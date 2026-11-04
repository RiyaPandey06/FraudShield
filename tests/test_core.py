from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd

from app.database import Database
from app.drift import DriftMonitor, psi
from app.evaluation import (
    best_cost_threshold, chronological_split, paired_bootstrap, precision_at_k, recall_at_precision, total_cost,
)
from app.explain import OllamaExplainer, facts_for, is_grounded, template_explanation
from app.features import FEATURE_NAMES, GRAPH_FEATURES, TABULAR_FEATURES, to_epoch
from app.graph import GraphService
from app.pipeline import compute_features
from app.synthetic import make_transactions

DELAY = 4 * 3600.0


# ---------------- data & features ----------------
def test_synthetic_is_deterministic_and_imbalanced(small_stream):
    assert small_stream == make_transactions(6000, seed=42)
    rate = np.mean([t["is_fraud"] for t in small_stream])
    assert 0.02 < rate < 0.05
    times = [to_epoch(t["timestamp"]) for t in small_stream]
    assert times == sorted(times)


def test_features_are_causal(small_stream, small_frame):
    """Features of the first N rows must not change when later events are appended."""
    prefix, _ = compute_features(small_stream[:2000], DELAY)
    full, _ = small_frame
    pd.testing.assert_frame_equal(prefix[FEATURE_NAMES], full.iloc[:2000][FEATURE_NAMES], check_exact=False, rtol=1e-9)


def test_label_maturity_is_enforced():
    g = GraphService()
    tx = {"transaction_id": "t1", "account_id": "a", "device_id": "d", "merchant_id": "m"}
    g.observe(tx)
    g.schedule_label(tx, 1, available_at=1000.0)
    assert g.release(999.0) == [] and g.context(tx)["account_risk"] == 0
    assert g.release(1000.0) == [("t1", 1)] and g.context(tx)["account_risk"] == 1


def test_graph_flags_shared_device_with_confirmed_fraud():
    g = GraphService()
    bad = {"transaction_id": "t0", "account_id": "bad", "device_id": "dev", "merchant_id": "m"}
    g.observe(bad)
    g.schedule_label(bad, 1, 0.0)
    for i in range(3):
        g.observe({"transaction_id": f"x{i}", "account_id": f"acc{i}", "device_id": "dev", "merchant_id": "m"})
    g.release(10.0)
    ctx = g.context({"transaction_id": "n", "account_id": "newcomer", "device_id": "dev", "merchant_id": "m"})
    assert ctx["neighbor_fraud_accounts"] == 1 and ctx["device_risk"] == 1 and ctx["mule_flag"] == 1


def test_train_serve_parity(bundle):
    """The live service must compute exactly the features the training pipeline would for the same stream."""
    for tx in bundle.live_stream:
        bundle.process({k: v for k, v in tx.items() if k not in {"is_fraud", "scenario"}})
    _, warm = compute_features(bundle.base_stream, DELAY)
    expected, _ = compute_features(bundle.live_stream, DELAY, state=warm)
    with bundle.database.connection() as conn:
        rows = conn.execute("SELECT transaction_id, features_json FROM transactions ORDER BY ts").fetchall()
    got = pd.DataFrame([json.loads(r["features_json"]) for r in rows])
    assert len(got) == len(expected)
    for col in FEATURE_NAMES:
        np.testing.assert_allclose(got[col].to_numpy(), expected[col].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=col)


# ---------------- evaluation ----------------
def test_split_is_chronological_with_embargo(small_frame):
    frame, _ = small_frame
    train, val, test = chronological_split(frame, embargo_seconds=DELAY)
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()
    # every kept training row is at least one label-delay older than the first validation row
    assert val["ts"].min() - train["ts"].max() >= DELAY - 1e-6
    assert len(train) < int(len(frame) * 0.70)       # some rows were purged


def test_metrics_and_cost_threshold_are_exact():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    scores = np.where(y == 1, rng.beta(5, 2, 500), rng.beta(2, 5, 500))
    thr, cost = best_cost_threshold(y, scores)
    brute = min(total_cost(y, scores, t) for t in np.unique(scores))
    assert cost <= brute + 1e-9 and total_cost(y, scores, thr) == cost
    assert recall_at_precision(y, y.astype(float)) == 1.0
    assert precision_at_k(y, y.astype(float), 0.05) == 1.0


def test_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.08).astype(int)
    weak = rng.normal(size=2000) + y * 0.5
    strong = rng.normal(size=2000) + y * 2.5
    res = paired_bootstrap(y, weak, strong, "pr_auc", n_boot=200)
    assert res["diff_lo"] > 0 and res["b"] > res["a"]


# ---------------- drift ----------------
def test_psi_stable_vs_shifted_and_binary():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 500)) < 0.1
    assert psi(ref, rng.normal(1.0, 1, 500)) > 0.2
    assert psi(np.zeros(1000), np.ones(500)) > 0.2                      # constant reference still detectable
    assert psi((rng.random(2000) < 0.02).astype(float), (rng.random(500) < 0.3).astype(float)) > 0.2


def test_drift_policy_needs_two_signals():
    rng = np.random.default_rng(0)
    cols = {n: rng.normal(size=2000).tolist() for n in ["amount_log", "amount_zscore", "velocity_10m"]}
    monitor = DriftMonitor(cols, rng.random(2000).tolist(), min_features=2, min_rows=100)
    stable = [{"amount_log": rng.normal(), "amount_zscore": rng.normal(), "velocity_10m": rng.normal(),
               "risk_score": rng.random()} for _ in range(300)]
    assert not monitor.evaluate(stable).triggered
    one = [{**r, "amount_log": r["amount_log"] + 3} for r in stable]
    two = [{**r, "amount_zscore": r["amount_zscore"] + 3} for r in one]
    assert not monitor.evaluate(one).triggered and monitor.evaluate(two).triggered
    assert monitor.evaluate(stable[:50]) is None                         # too few rows


# ---------------- model ----------------
def test_model_scores_and_reasons(bundle):
    model = bundle.model.model
    row = pd.DataFrame([{n: 0.0 for n in FEATURE_NAMES}])
    assert model.explainer in {"treeshap", "occlusion"}
    score, contrib = model.score_with_contributions(row.to_numpy()[0])
    assert 0.0 <= score <= 1.0 and len(contrib) == len(FEATURE_NAMES)
    if model.explainer == "treeshap":
        assert abs(score - model.predict_proba(row)[0]) < 1e-6


def test_graph_features_improve_over_tabular_only(small_frame):
    """Sanity check of the ablation machinery on the small fixture: the graph model must not be worse by a lot."""
    from app.model import GradientBoostedModel
    frame, _ = small_frame
    train, val, test = chronological_split(frame, embargo_seconds=DELAY)
    from app.evaluation import pr_auc
    tab = GradientBoostedModel(TABULAR_FEATURES).fit(train, train["label"].to_numpy())
    allf = GradientBoostedModel(FEATURE_NAMES).fit(train, train["label"].to_numpy())
    y = test["label"].to_numpy()
    assert pr_auc(y, allf.predict_proba(test)) >= pr_auc(y, tab.predict_proba(test)) - 0.05
    assert set(GRAPH_FEATURES).isdisjoint(TABULAR_FEATURES)


def test_promotion_gate_rejects_a_worse_candidate(bundle):
    from app.services import load_base_frame
    ms = bundle.model
    version_before = ms.metadata["model_version"]
    base = load_base_frame(bundle.settings)
    rng = np.random.default_rng(0)
    early = base.iloc[: len(base) // 2]
    poison = early.sample(frac=0.8, random_state=1).copy()
    poison["label"] = rng.integers(0, 2, len(poison))        # random labels, timestamps only in the training window
    result = ms.train(base, poison)
    assert result["promotion"]["promoted"] is False
    assert ms.metadata["model_version"] == version_before
    assert json.loads(bundle.settings.champion_path.read_text())["version"] == version_before
    again = ms.train(base, None)                              # an equally good candidate is promoted
    assert again["promotion"]["promoted"] is True and ms.metadata["model_version"] == again["model_version"]


def test_default_threshold_is_the_cost_tuned_one(bundle):
    assert bundle.threshold == bundle.model.metadata["metrics"]["cost_threshold"]


# ---------------- service behaviour ----------------
def test_service_records_labels_drift_and_alert_rows(bundle):
    summary = bundle.database.summary()
    assert summary["transactions_processed"] >= 600 and summary["labels_matured"] > 0
    assert bundle.database.list_alerts(limit=5)[0]["reasons"]
    assert bundle.database.latest_drift() is not None


def test_retrain_defers_without_enough_labels_and_runs_with_them(bundle):
    bundle.settings = type(bundle.settings)(**{**bundle.settings.__dict__, "retrain_min_labels": 10**6})
    assert bundle.retrain("drift")["outcome"] == "deferred"
    bundle.settings = type(bundle.settings)(**{**bundle.settings.__dict__, "retrain_min_labels": 50})
    assert bundle.retrain("manual")["outcome"] in {"promoted", "rejected"}
    assert bundle.database.retrain_history()


# ---------------- database ----------------
def test_feedback_overrides_matured_label(bundle):
    alert = bundle.database.list_alerts(limit=1)[0]
    assert bundle.database.record_feedback(alert["transaction_id"], "legitimate", "t", "", 4200, "raw")
    rows = {r["transaction_id"]: r for r in bundle.database.labeled_live_rows()}
    assert rows[alert["transaction_id"]]["label"] == 0
    assert bundle.database.review_rows()[-1] == {"arm": "raw", "review_ms": 4200}
    assert not bundle.database.record_feedback("missing", "fraud", "t", "")


# ---------------- explanations ----------------
ALERT = {"risk_score": 0.87, "amount": 540.0, "country": "IN", "graph": {"linked_accounts": 3, "community_size": 9},
         "reasons": [{"feature": "device_account_count", "value": 3.0, "contribution": 1.2, "direction": "increases_risk",
                      "label": "device is shared by several accounts"}]}


def test_template_always_available():
    text = template_explanation(0.87, ALERT["reasons"], ALERT["graph"], "alert")
    assert "87%" in text and "device is shared" in text


def test_grounding_rejects_invented_numbers():
    facts = facts_for(ALERT)
    assert is_grounded("Risk is 87% and the device links 3 accounts.", facts)
    assert not is_grounded("This card was used 14 times today.", facts)


def _mock_ollama(reply: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"response": reply}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):  # silence
            pass
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_ollama_client_accepts_grounded_and_rejects_hallucinated_text():
    good = _mock_ollama("Flagged at 87% because the device is shared across 3 accounts.")
    bad = _mock_ollama("Flagged because this card made 14 purchases in Lagos.")
    try:
        assert OllamaExplainer(f"http://127.0.0.1:{good.server_port}", "m").explain(ALERT).startswith("Flagged at 87%")
        assert OllamaExplainer(f"http://127.0.0.1:{bad.server_port}", "m").explain(ALERT) is None
    finally:
        good.shutdown(); bad.shutdown()
    assert OllamaExplainer("http://127.0.0.1:9", "m", timeout=1).explain(ALERT) is None   # LLM down -> fallback


def test_llm_upgrade_flow_in_database(tmp_path):
    db = Database(tmp_path / "x.db")
    db.initialize()
    tx = {"transaction_id": "t1", "timestamp": "2026-01-01T00:00:00+00:00", "account_id": "a", "device_id": "d",
          "merchant_id": "m", "amount": 5.0, "country": "IN", "transaction_type": "card_purchase"}
    result = {"decision": "alert", "threshold": 0.1, "model_version": "v", "reasons": ALERT["reasons"],
              "graph_context": ALERT["graph"], "explanation": "template text"}
    db.save_scored_transaction(tx, 0.0, {"f": 1.0}, 0.9, result)
    assert db.alerts_needing_llm()[0]["transaction_id"] == "t1"
    db.update_explanation("t1", "llm text", "llm")
    assert db.get_alert("t1")["explanation_source"] == "llm" and not db.alerts_needing_llm()

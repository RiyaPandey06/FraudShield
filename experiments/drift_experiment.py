"""Drift lead-time experiment.

Timeline (all measured on a continuous, time-ordered stream with delayed labels):
  * t_psi      - first monitoring check that fires the retraining trigger (PSI on inputs/scores; no labels needed)
  * t_oracle   - first check where PR-AUC (computed with *instant*, unrealistic labels) has dropped
  * t_measured - first check where PR-AUC computed only from *matured* labels shows a sustained drop
Lead time = t_measured - t_psi: how much earlier the label-free monitor warns than a label-based one could.
A drift-free control stream measures false alarms. Retraining is then tried at several delays after the trigger.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.drift import DriftMonitor
from app.evaluation import best_cost_threshold, pr_auc, summarise
from app.features import FEATURE_NAMES
from app.model import GradientBoostedModel
from app.pipeline import compute_features
from app.synthetic import BASE_START, DriftSpec, make_transactions

DELAY = 4 * 3600.0
CHECK_EVERY, WINDOW, PERF_WINDOW, DROP = 100, 500, 1500, 0.10


def _train_champion(base: pd.DataFrame):
    n = len(base)
    train = base.iloc[: int(n * 0.70)]
    train = train[train["ts"] <= train["ts"].max() - DELAY]
    val = base.iloc[int(n * 0.70): int(n * 0.85)]
    model = GradientBoostedModel(FEATURE_NAMES, seed=42).fit(train, train["label"].to_numpy())
    thr, _ = best_cost_threshold(val["label"].to_numpy(), model.predict_proba(val))
    test = base.iloc[int(n * 0.85):]
    recent = train.iloc[int(len(train) * 0.6):]
    sample = recent.sample(min(2000, len(recent)), random_state=42)
    monitor = DriftMonitor({c: sample[c].tolist() for c in FEATURE_NAMES}, model.predict_proba(sample).tolist(),
                           psi_threshold=0.2, min_features=2, min_rows=WINDOW)
    return model, thr, monitor, pr_auc(test["label"].to_numpy(), model.predict_proba(test))


def monitor_stream(live: pd.DataFrame, scores: np.ndarray, monitor: DriftMonitor) -> list[dict]:
    y, ts = live["label"].to_numpy(), live["ts"].to_numpy()
    feats = live[FEATURE_NAMES].assign(risk_score=scores)
    checks = []
    for n in range(WINDOW, len(live) + 1, CHECK_EVERY):
        window = feats.iloc[n - WINDOW: n].to_dict("records")
        report = monitor.evaluate(window)
        now = ts[n - 1]
        matured = np.flatnonzero(ts[:n] + DELAY <= now)[-PERF_WINDOW:]
        oracle = slice(max(0, n - PERF_WINDOW), n)
        checks.append({
            "n": n, "ts": float(now), "triggered": bool(report.triggered), "drifted": report.drifted,
            "max_psi": max(r["psi"] for r in report.results),
            "pr_auc_measured": pr_auc(y[matured], scores[matured]) if len(matured) > 300 and y[matured].sum() >= 15 else None,
            "pr_auc_oracle": pr_auc(y[oracle], scores[oracle]) if y[oracle].sum() >= 15 else None,
        })
    return checks


def _first_sustained(checks: list[dict], key: str, reference: float, after_n: int, k: int = 2) -> dict | None:
    """First check (after drift could have started) where `key` is >DROP below reference for k consecutive checks."""
    run = 0
    for i, c in enumerate(checks):
        if c["n"] < after_n or c[key] is None:
            run = 0
            continue
        run = run + 1 if c[key] < reference * (1 - DROP) else 0
        if run >= k:
            return checks[i - k + 1]
    return None


def _retrain_at(all_live: pd.DataFrame, base: pd.DataFrame, t_retrain: float):
    pool = pd.concat([base, all_live[all_live["ts"] + DELAY <= t_retrain]], ignore_index=True).sort_values("ts")
    n = len(pool)
    train = pool.iloc[: int(n * 0.85)]
    train = train[train["ts"] <= train["ts"].max() - DELAY]
    val = pool.iloc[int(n * 0.85):]
    val = val[val["ts"] <= val["ts"].max() - DELAY]
    model = GradientBoostedModel(FEATURE_NAMES, seed=42).fit(train, train["label"].to_numpy())
    thr, _ = best_cost_threshold(val["label"].to_numpy(), model.predict_proba(val))
    return model, thr, len(pool) - len(base)


def run(base_rows: int = 40_000, live_rows: int = 24_000, drift_start: int = 8_000, ramp: int = 4_000) -> dict:
    started = time.time()
    base_stream = make_transactions(base_rows, seed=42)
    base, state = compute_features(base_stream, DELAY)
    champion, thr, monitor, ref_pr = _train_champion(base)
    base_end = datetime.fromtimestamp(int(base["ts"].max()), tz=UTC)
    start = base_end + timedelta(hours=1)
    streams = {
        "drift": make_transactions(live_rows, seed=73, world_seed=42, start=start, id_prefix="live",
                                   drift=DriftSpec(start_index=drift_start, ramp_rows=ramp)),
        "control": make_transactions(live_rows, seed=73, world_seed=42, start=start, id_prefix="ctrl"),
    }
    out: dict = {"base_rows": base_rows, "live_rows": live_rows, "drift_start_index": drift_start, "ramp_rows": ramp,
                 "label_delay_hours": DELAY / 3600, "backend": champion.backend, "reference_test_pr_auc": round(ref_pr, 4),
                 "check_every": CHECK_EVERY, "window": WINDOW, "psi_threshold": 0.2, "min_drifted_features": 2}
    frames, checks = {}, {}
    for name, stream in streams.items():
        frame, _ = compute_features(stream, DELAY, state=copy.deepcopy(state))
        frames[name] = frame
        scores = champion.predict_proba(frame)
        checks[name] = monitor_stream(frame, scores, monitor)

    ctrl, drift = checks["control"], checks["drift"]
    out["control"] = {"checks": len(ctrl), "triggered_checks": sum(c["triggered"] for c in ctrl),
                      "max_psi": round(max(c["max_psi"] for c in ctrl), 4)}
    pre = [c for c in drift if c["n"] <= drift_start]
    out["pre_drift_false_alarm_checks"] = sum(c["triggered"] for c in pre)
    pre_measured = [c["pr_auc_measured"] for c in pre if c["pr_auc_measured"] is not None]
    live_base = float(np.mean(pre_measured)) if pre_measured else ref_pr
    out["live_pre_drift_pr_auc"] = round(live_base, 4)

    t_psi = next((c for c in drift if c["triggered"] and c["n"] > drift_start), None)
    t_oracle = _first_sustained(drift, "pr_auc_oracle", live_base, drift_start)
    t_meas = _first_sustained(drift, "pr_auc_measured", live_base, drift_start)
    hours = lambda c: None if c is None else round((c["ts"] - drift[0]["ts"]) / 3600, 1)  # noqa: E731
    out["detection"] = {
        "t_psi_row": None if t_psi is None else t_psi["n"], "t_oracle_drop_row": None if t_oracle is None else t_oracle["n"],
        "t_measured_drop_row": None if t_meas is None else t_meas["n"],
        "psi_rows_after_drift_start": None if t_psi is None else t_psi["n"] - drift_start,
        "psi_features": None if t_psi is None else t_psi["drifted"],
        "lead_time_rows_vs_measured": None if not (t_psi and t_meas) else t_meas["n"] - t_psi["n"],
        "lead_time_hours_vs_measured": None if not (t_psi and t_meas) else round((t_meas["ts"] - t_psi["ts"]) / 3600, 1),
        "lead_time_rows_vs_oracle": None if not (t_psi and t_oracle) else t_oracle["n"] - t_psi["n"],
        "hours_psi": hours(t_psi),
    }
    out["timeline"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != "ts"}
                       for c in drift[:: 2]]

    # --- retraining experiment: evaluate everything on the final, fully-drifted 6000 rows ----------
    live = frames["drift"]
    test_start = len(live) - 6_000
    test = live.iloc[test_start:]
    y = test["label"].to_numpy()
    results = {"no_retrain": summarise(y, champion.predict_proba(test), thr)}
    if t_psi is not None:
        for label, hrs in (("retrain_at_trigger", 0), ("retrain_trigger_plus_24h", 24), ("retrain_trigger_plus_72h", 72), ("retrain_trigger_plus_120h", 120)):
            t_re = t_psi["ts"] + hrs * 3600
            if t_re >= test["ts"].iloc[0]:
                continue
            model, new_thr, n_live = _retrain_at(live.iloc[:test_start], base, t_re)
            results[label] = {**summarise(y, model.predict_proba(test), new_thr), "live_labeled_rows_used": int(n_live)}
    out["retraining"] = results
    out["seconds"] = round(time.time() - started, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-rows", type=int, default=40_000)
    parser.add_argument("--live-rows", type=int, default=24_000)
    parser.add_argument("--out", default="results/drift.json")
    args = parser.parse_args()
    result = run(args.base_rows, args.live_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "timeline"}, indent=2))


if __name__ == "__main__":
    main()

"""Graph-feature ablation: logistic regression vs GBM(tabular) vs GBM(tabular+graph).

Strict chronological split with a label-maturity embargo; thresholds are chosen on validation only;
CIs come from a paired bootstrap over the test rows; seed variance from repeated training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.evaluation import (
    best_cost_threshold, chronological_split, paired_bootstrap, summarise,
)
from app.features import FEATURE_NAMES, GRAPH_FEATURES, TABULAR_FEATURES
from app.model import GradientBoostedModel, baseline_scores, fit_baseline
from app.pipeline import compute_features
from app.synthetic import make_transactions

LABEL_DELAY = 4 * 3600.0


def run(rows: int = 60_000, seeds: int = 5, n_boot: int = 1000, data_seed: int = 42) -> dict:
    started = time.time()
    frame, _ = compute_features(make_transactions(rows=rows, seed=data_seed), LABEL_DELAY)
    train, val, test = chronological_split(frame, embargo_seconds=LABEL_DELAY)
    y_tr, y_val, y_te = train["label"].to_numpy(), val["label"].to_numpy(), test["label"].to_numpy()
    out: dict = {"rows": rows, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                 "test_frauds": int(y_te.sum()), "label_delay_hours": LABEL_DELAY / 3600, "seeds": seeds,
                 "backend": GradientBoostedModel(TABULAR_FEATURES).backend, "per_seed": [], "n_boot": n_boot}

    lr = fit_baseline(train, y_tr)
    lr_val, lr_te = baseline_scores(lr, val), baseline_scores(lr, test)
    lr_thr, _ = best_cost_threshold(y_val, lr_val)
    out["logistic_regression"] = summarise(y_te, lr_te, lr_thr)

    tab_scores, all_scores = [], []
    for seed in range(seeds):
        row = {"seed": seed}
        for name, cols, sink in (("gbm_tabular", TABULAR_FEATURES, tab_scores), ("gbm_tabular_graph", FEATURE_NAMES, all_scores)):
            model = GradientBoostedModel(cols, seed=seed).fit(train, y_tr)
            s_val, s_te = model.predict_proba(val), model.predict_proba(test)
            thr, _ = best_cost_threshold(y_val, s_val)
            row[name] = summarise(y_te, s_te, thr)
            row[f"{name}_threshold"] = thr
            sink.append((s_te, thr))
        out["per_seed"].append(row)

    for name in ("gbm_tabular", "gbm_tabular_graph"):
        out[f"{name}_mean"] = {k: round(float(np.mean([r[name][k] for r in out["per_seed"]])), 4)
                               for k in out["per_seed"][0][name]}
        out[f"{name}_std"] = {k: round(float(np.std([r[name][k] for r in out["per_seed"]])), 4)
                              for k in ("pr_auc", "recall_at_90_precision", "precision_at_1pct", "cost_at_threshold")}

    # Paired bootstrap on seed 0 (same test rows resampled for both models).
    boot = {m: paired_bootstrap(y_te, tab_scores[0][0], all_scores[0][0], m, n_boot=n_boot)
            for m in ("pr_auc", "recall_at_90_precision", "precision_at_1pct")}
    boot_lr = {m: paired_bootstrap(y_te, lr_te, tab_scores[0][0], m, n_boot=n_boot) for m in ("pr_auc",)}
    out["bootstrap_graph_vs_tabular"], out["bootstrap_gbm_vs_lr"] = boot, boot_lr

    # Where does the graph help? Recall at each model's own cost threshold, by fraud scenario.
    scen = {}
    for label, (scores, thr) in (("gbm_tabular", tab_scores[0]), ("gbm_tabular_graph", all_scores[0])):
        for kind in ("solo", "ring", "stealth"):
            mask = (test["scenario"] == kind).to_numpy()
            scen.setdefault(kind, {})[label] = round(float((scores[mask] >= thr).mean()), 4) if mask.any() else None
            scen[kind]["n"] = int(mask.sum())
    out["recall_by_scenario"] = scen
    out["graph_features"] = GRAPH_FEATURES
    out["seconds"] = round(time.time() - started, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--out", default="results/ablation.json")
    args = parser.parse_args()
    result = run(args.rows, args.seeds, args.boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("backend", "test_rows", "test_frauds", "logistic_regression",
                                            "gbm_tabular_mean", "gbm_tabular_graph_mean",
                                            "bootstrap_graph_vs_tabular", "recall_by_scenario")}, indent=2))


if __name__ == "__main__":
    main()

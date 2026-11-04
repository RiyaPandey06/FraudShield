"""Imbalance-aware evaluation: strict time splits with a label-maturity embargo, PR metrics, cost, bootstrap."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

FN_COST, FP_COST = 500.0, 20.0


def chronological_split(frame: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15,
                        embargo_seconds: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val/test. Rows whose label would not yet be mature at the next boundary
    (``ts > boundary - embargo``) are purged from the earlier segment, mirroring delayed labels in production."""
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(frame)
    i_train, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    t_train_end, t_val_end = frame["ts"].iloc[i_train], frame["ts"].iloc[i_val]
    train = frame.iloc[:i_train]
    val = frame.iloc[i_train:i_val]
    train = train[train["ts"] <= t_train_end - embargo_seconds]
    val = val[val["ts"] <= t_val_end - embargo_seconds]
    return train, val, frame.iloc[i_val:]


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y, scores)) if y.sum() else float("nan")


def recall_at_precision(y: np.ndarray, scores: np.ndarray, target: float = 0.90) -> float:
    if not y.sum():
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, scores)
    viable = precision[:-1] >= target
    return float(recall[:-1][viable].max()) if viable.any() else 0.0


def precision_at_k(y: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float:
    k = max(1, round(len(y) * fraction))
    return float(y[np.argsort(-scores, kind="stable")[:k]].mean())


def best_cost_threshold(y: np.ndarray, scores: np.ndarray, fn_cost: float = FN_COST,
                        fp_cost: float = FP_COST) -> tuple[float, float]:
    """Exact minimiser of fn_cost*FN + fp_cost*FP over all cut points (flag when score >= threshold)."""
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    costs = fn_cost * (ys.sum() - tp) + fp_cost * fp
    # only cut between distinct scores
    valid = np.append(ss[1:] < ss[:-1], True)
    costs_valid = np.where(valid, costs, np.inf)
    best = int(np.argmin(costs_valid))
    no_flag_cost = fn_cost * ys.sum()
    if no_flag_cost <= costs_valid[best]:
        return float(ss[0]) + 1e-9, float(no_flag_cost)
    return float(ss[best]), float(costs_valid[best])


def total_cost(y: np.ndarray, scores: np.ndarray, threshold: float, fn_cost: float = FN_COST,
               fp_cost: float = FP_COST) -> float:
    flagged = scores >= threshold
    return float(fn_cost * ((y == 1) & ~flagged).sum() + fp_cost * ((y == 0) & flagged).sum())


def summarise(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores >= threshold
    tp = int(((y == 1) & flagged).sum())
    return {
        "pr_auc": round(pr_auc(y, scores), 4),
        "recall_at_90_precision": round(recall_at_precision(y, scores), 4),
        "precision_at_1pct": round(precision_at_k(y, scores, 0.01), 4),
        "precision_at_threshold": round(tp / max(int(flagged.sum()), 1), 4),
        "recall_at_threshold": round(tp / max(int(y.sum()), 1), 4),
        "cost_at_threshold": total_cost(y, scores, threshold),
        "flag_nothing_cost": float(FN_COST * y.sum()),
    }


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "pr_auc": pr_auc,
    "recall_at_90_precision": recall_at_precision,
    "precision_at_1pct": precision_at_k,
}


def paired_bootstrap(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, metric: str,
                     n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Resample test rows with replacement; return CIs for A, B and the paired difference (B - A)."""
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    a_vals, b_vals = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        a_vals.append(fn(y[idx], scores_a[idx]))
        b_vals.append(fn(y[idx], scores_b[idx]))
    a_arr, b_arr = np.array(a_vals), np.array(b_vals)
    diff = b_arr - a_arr
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "a": round(fn(y, scores_a), 4), "a_lo": round(float(np.percentile(a_arr, 2.5)), 4),
        "a_hi": round(float(np.percentile(a_arr, 97.5)), 4),
        "b": round(fn(y, scores_b), 4), "b_lo": round(float(np.percentile(b_arr, 2.5)), 4),
        "b_hi": round(float(np.percentile(b_arr, 97.5)), 4),
        "diff": round(float(diff.mean()), 4), "diff_lo": round(float(lo), 4), "diff_hi": round(float(hi), 4),
        "p_b_not_better": round(float((diff <= 0).mean()), 4),
    }

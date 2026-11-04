"""Analyst review-time study: raw SHAP reasons only (arm 'raw') vs reasons + explanation text (arm 'explained').

The dashboard randomly assigns an arm per case (hash of the transaction id) and records milliseconds from
opening a case to submitting a label. This script turns those REAL recorded timings into a percent change
with a bootstrap CI. It never fabricates data: with fewer than --min-per-arm cases it reports 'insufficient data'.
Run it after real people (even 2-3 teammates) have labelled 30+ cases per arm.
"""
from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np


def analyse(rows: list[dict], min_per_arm: int = 20, n_boot: int = 5000, seed: int = 0) -> dict:
    raw = np.array([r["review_ms"] for r in rows if r["arm"] == "raw"], float)
    exp = np.array([r["review_ms"] for r in rows if r["arm"] == "explained"], float)
    if len(raw) < min_per_arm or len(exp) < min_per_arm:
        return {"status": "insufficient data", "raw_n": len(raw), "explained_n": len(exp), "needed_per_arm": min_per_arm}
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(n_boot):
        a, b = rng.choice(raw, len(raw)), rng.choice(exp, len(exp))
        changes.append((np.median(b) - np.median(a)) / np.median(a) * 100)
    lo, hi = np.percentile(changes, [2.5, 97.5])
    return {"status": "ok", "raw_n": len(raw), "explained_n": len(exp),
            "raw_median_s": round(float(np.median(raw)) / 1000, 2), "explained_median_s": round(float(np.median(exp)) / 1000, 2),
            "median_change_pct": round(float((np.median(exp) - np.median(raw)) / np.median(raw) * 100), 1),
            "ci95_pct": [round(float(lo), 1), round(float(hi), 1)],
            "note": "Negative = faster with explanations. Median-based; bootstrap over cases."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/fraudshield.db")
    parser.add_argument("--min-per-arm", type=int, default=20)
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT arm, review_ms FROM feedback WHERE arm IS NOT NULL AND review_ms IS NOT NULL")]
    print(json.dumps(analyse(rows, args.min_per_arm), indent=2))


if __name__ == "__main__":
    main()

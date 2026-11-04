from __future__ import annotations

import numpy as np

from experiments.review_time_report import analyse


def test_insufficient_data_is_reported_not_invented():
    assert analyse([{"arm": "raw", "review_ms": 9000}] * 5)["status"] == "insufficient data"


def test_detects_a_real_speedup():
    rng = np.random.default_rng(0)
    rows = [{"arm": "raw", "review_ms": float(rng.normal(30000, 3000))} for _ in range(60)]
    rows += [{"arm": "explained", "review_ms": float(rng.normal(21000, 3000))} for _ in range(60)]
    res = analyse(rows)
    assert res["status"] == "ok" and res["ci95_pct"][1] < 0 and -40 < res["median_change_pct"] < -20

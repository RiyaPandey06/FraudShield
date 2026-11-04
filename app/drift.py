"""Drift monitoring: PSI + KS per feature (and on the score) with an explicit retraining trigger.

Evidently is supported as an optional HTML report (``evidently_report``); the trigger itself uses
PSI so it has no heavy dependency and runs inside the scoring path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

MONITORED = ["amount_log", "amount_zscore", "velocity_10m", "velocity_1h", "merchant_frequency",
             "new_device", "country_new", "device_account_count", "risk_score"]


def psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Population Stability Index. Low-cardinality features use one bin per value, others use deciles."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if not len(reference) or not len(current):
        return 0.0
    values = np.unique(reference)
    if len(values) <= 10:
        mids = (values[:-1] + values[1:]) / 2
        edges = np.concatenate([[-np.inf], mids if len(values) > 1 else [values[0] + 1e-9], [np.inf]])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 2:
        return 0.0
    expected, _ = np.histogram(reference, bins=edges)
    actual, _ = np.histogram(current, bins=edges)
    expected = np.clip(expected / expected.sum(), 1e-4, None)
    actual = np.clip(actual / max(actual.sum(), 1), 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@dataclass
class DriftReport:
    results: list[dict]
    drifted: list[str]
    triggered: bool


class DriftMonitor:
    def __init__(self, reference: dict[str, list[float]], reference_scores: list[float],
                 psi_threshold: float = 0.20, min_features: int = 2, min_rows: int = 200) -> None:
        self.reference = {k: np.asarray(v, float) for k, v in reference.items()}
        self.reference["risk_score"] = np.asarray(reference_scores, float)
        self.psi_threshold, self.min_features, self.min_rows = psi_threshold, min_features, min_rows

    @classmethod
    def from_metadata(cls, metadata: dict, **kwargs) -> DriftMonitor:
        ref = metadata["reference"]
        return cls(ref["features"], ref["scores"], **kwargs)

    def evaluate(self, window: list[dict]) -> DriftReport | None:
        """``window`` is a list of feature dicts that include ``risk_score``."""
        if len(window) < self.min_rows:
            return None
        results, drifted = [], []
        for name in MONITORED:
            if name not in self.reference:
                continue
            current = np.array([row[name] for row in window], float)
            value = psi(self.reference[name], current)
            ks = ks_2samp(self.reference[name], current)
            status = "drift" if value >= self.psi_threshold else "stable"
            if status == "drift":
                drifted.append(name)
            results.append({"feature_name": name, "psi": round(value, 4), "ks_stat": round(float(ks.statistic), 4),
                            "status": status})
        return DriftReport(results, drifted, len(drifted) >= self.min_features)


def evidently_report(reference: dict[str, list[float]], current_rows: list[dict], path: str) -> str | None:
    """Optional: write an Evidently data-drift HTML report. Returns the path, or None if Evidently is absent."""
    try:
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset  # evidently 0.4.x API
        from evidently.report import Report
    except Exception:  # noqa: BLE001 - optional dependency / API differences across versions
        return None
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=pd.DataFrame(reference), current_data=pd.DataFrame(current_rows)[list(reference)])
    report.save_html(path)
    return path

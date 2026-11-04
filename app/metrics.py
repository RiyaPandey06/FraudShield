"""Prometheus metrics. Falls back to no-ops when prometheus_client is not installed (e.g. in unit tests)."""
from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, start_http_server
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def __init__(self, *_, **__): ...
        def labels(self, *_, **__): return self
        def inc(self, *_): ...
        def set(self, *_): ...
        def observe(self, *_): ...

    Counter = Gauge = Histogram = _Noop

    def generate_latest() -> bytes:
        return b"# prometheus_client not installed\n"

    def start_http_server(*_, **__): ...

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

SCORED = Counter("fraudshield_transactions_scored_total", "Transactions scored", ["decision"])
LATENCY = Histogram("fraudshield_scoring_latency_seconds", "End-to-end scoring latency", buckets=BUCKETS)
ALERT_RATE = Gauge("fraudshield_alert_rate", "Share of alerts in the last 500 scored transactions")
MODEL_INFO = Gauge("fraudshield_model_info", "Currently served model", ["version", "backend"])
DRIFT_PSI = Gauge("fraudshield_drift_psi", "Latest PSI per monitored feature", ["feature"])
DRIFT_TRIGGERS = Counter("fraudshield_drift_triggers_total", "Drift events that triggered retraining")
RETRAINS = Counter("fraudshield_retrains_total", "Retraining runs", ["outcome"])
FEEDBACK = Counter("fraudshield_feedback_total", "Analyst labels received", ["label"])
LABELS_MATURED = Counter("fraudshield_labels_matured_total", "Delayed labels that matured")

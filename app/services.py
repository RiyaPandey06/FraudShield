from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import joblib
import pandas as pd

from app import metrics
from app.config import Settings
from app.database import Database
from app.datasets import load_base_stream
from app.drift import DriftMonitor, DriftReport
from app.explain import template_explanation
from app.model import ModelService
from app.pipeline import StreamState, compute_features

log = logging.getLogger("fraudshield.services")


def load_base_frame(settings: Settings) -> pd.DataFrame:
    """Labelled feature history, cached on the shared volume (computed once by the init container)."""
    if settings.base_frame_path.exists():
        return joblib.load(settings.base_frame_path)
    frame, _ = compute_features(load_base_stream(settings), settings.label_delay_seconds)
    joblib.dump(frame, settings.base_frame_path)
    return frame


def load_label_feed(settings: Settings) -> dict[str, int]:
    """Simulated delayed-label feed (chargebacks). Truth is only *released* after the maturity delay."""
    path = settings.data_dir / "demo_labels.jsonl"
    if not path.exists():
        return {}
    feed = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            feed[item["transaction_id"]] = int(item["label"])
    return feed


@dataclass
class ServiceBundle:
    settings: Settings
    database: Database
    state: StreamState
    model: ModelService
    drift: DriftMonitor
    label_feed: dict[str, int] = field(default_factory=dict)
    window: deque = field(default_factory=lambda: deque(maxlen=500))
    recent_decisions: deque = field(default_factory=lambda: deque(maxlen=500))
    processed: int = 0
    last_trigger_at: int = -10**9
    _retraining: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, settings: Settings, warm_start: bool = True, label_feed: dict[str, int] | None = None) -> ServiceBundle:
        database = Database(settings.database_path)
        database.initialize()
        model = ModelService(settings)
        model.ensure_model(lambda: load_base_frame(settings))
        state = StreamState(label_delay=settings.label_delay_seconds)
        if warm_start:
            _, state = compute_features(load_base_stream(settings), settings.label_delay_seconds)
        bundle = cls(settings, database, state, model, cls._monitor(settings, model.metadata),
                     label_feed if label_feed is not None else load_label_feed(settings))
        bundle._publish_model_info()
        return bundle

    @staticmethod
    def _monitor(settings: Settings, metadata: dict) -> DriftMonitor:
        return DriftMonitor.from_metadata(metadata, psi_threshold=settings.drift_psi_threshold,
                                          min_features=settings.drift_min_features, min_rows=settings.drift_min_rows)

    def _publish_model_info(self) -> None:
        metrics.MODEL_INFO.labels(self.model.metadata["model_version"], self.model.metadata["backend"]).set(1)

    @property
    def threshold(self) -> float:
        return self.settings.threshold if self.settings.threshold is not None else self.model.metadata["metrics"]["cost_threshold"]

    # ---- scoring ----------------------------------------------------------
    def process(self, tx: dict) -> dict:
        started = time.perf_counter()
        if self.processed % 50 == 0 and self.model.reload_if_changed():
            self.drift = self._monitor(self.settings, self.model.metadata)
            self._publish_model_info()
        features, ctx, ts, released = self.state.featurize(tx)
        t_features = time.perf_counter()
        score, reasons = self.model.predict(features)
        t_model = time.perf_counter()
        threshold = self.threshold
        decision = "alert" if score >= threshold else "allow"
        result = {
            "transaction_id": tx["transaction_id"], "risk_score": round(score, 6), "decision": decision,
            "threshold": round(threshold, 6), "model_version": self.model.metadata["model_version"],
            "reasons": reasons, "graph_context": ctx,
            "explanation": template_explanation(score, reasons, ctx, decision),
        }
        self.database.save_scored_transaction(tx, ts, features, score, result)
        if released:
            self.database.save_labels(released, ts)
            metrics.LABELS_MATURED.inc(len(released))
        self.state.commit(tx, ts, self.label_feed.get(tx["transaction_id"]))
        t_db = time.perf_counter()
        self.processed += 1
        self.window.append({**features, "risk_score": score})
        self.recent_decisions.append(decision == "alert")
        result["latency_ms"] = round((t_db - started) * 1000, 3)
        result["timing_ms"] = {"features": round((t_features - started) * 1000, 3),
                               "model": round((t_model - t_features) * 1000, 3),
                               "database": round((t_db - t_model) * 1000, 3)}
        metrics.SCORED.labels(decision).inc()
        metrics.LATENCY.observe(t_db - started)
        metrics.ALERT_RATE.set(sum(self.recent_decisions) / len(self.recent_decisions))
        if self.processed % self.settings.drift_check_every == 0:
            self.check_drift(ts)
        return result

    # ---- drift & retraining --------------------------------------------------
    def check_drift(self, stream_ts: float) -> DriftReport | None:
        report = self.drift.evaluate(list(self.window)[-self.settings.drift_window:])
        if report is None:
            return None
        for item in report.results:
            metrics.DRIFT_PSI.labels(item["feature_name"]).set(item["psi"])
        cooled_down = self.processed - self.last_trigger_at >= self.settings.drift_cooldown_rows
        trigger = report.triggered and cooled_down
        self.database.record_drift_check(stream_ts, self.processed, report.results, report.drifted, trigger)
        if trigger:
            self.last_trigger_at = self.processed
            metrics.DRIFT_TRIGGERS.inc()
            log.warning("Drift detected on %s -> retraining trigger", report.drifted)
            if self.settings.auto_retrain:
                threading.Thread(target=self.retrain, args=("drift",), daemon=True).start()
        return report

    def retrain(self, trigger: str = "manual") -> dict:
        """Retrain on base history + matured live labels (+ analyst feedback); promote only if not worse."""
        if not self._retraining.acquire(blocking=False):
            return {"outcome": "skipped", "reason": "a retraining run is already in progress"}
        try:
            live = self.database.labeled_live_rows()
            if trigger == "drift" and len(live) < self.settings.retrain_min_labels:
                details = {"live_labeled_rows": len(live), "needed": self.settings.retrain_min_labels}
                self.database.log_retrain(trigger, "deferred", details)
                metrics.RETRAINS.labels("deferred").inc()
                self.last_trigger_at = -10**9   # allow the next drift check to re-trigger once labels mature
                return {"outcome": "deferred", **details}
            extra = pd.DataFrame(live) if live else None
            metadata = self.model.train(load_base_frame(self.settings), extra)
            promoted = metadata["promotion"]["promoted"]
            if promoted:
                self.drift = self._monitor(self.settings, metadata)
                self._publish_model_info()
            outcome = "promoted" if promoted else "rejected"
            details = {"model_version": metadata["model_version"], "live_labeled_rows": len(live),
                       "promotion": metadata["promotion"], "pr_auc": metadata["metrics"]["pr_auc"]}
            self.database.log_retrain(trigger, outcome, details)
            metrics.RETRAINS.labels(outcome).inc()
            return {"outcome": outcome, **details}
        except Exception as exc:  # noqa: BLE001
            self.database.log_retrain(trigger, "failed", {"error": str(exc)})
            metrics.RETRAINS.labels("failed").inc()
            log.exception("retraining failed")
            return {"outcome": "failed", "error": str(exc)}
        finally:
            self._retraining.release()

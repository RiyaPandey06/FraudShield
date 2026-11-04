"""One feature pipeline shared by training, experiments and live serving (guarantees train/serve parity)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.features import FEATURE_NAMES, FeatureService, to_epoch
from app.graph import GraphService


@dataclass
class StreamState:
    label_delay: float = 4 * 3600.0
    refresh_every: int = 500
    features: FeatureService = field(default_factory=FeatureService)
    graph: GraphService = field(init=False)

    def __post_init__(self) -> None:
        self.graph = GraphService(refresh_every=self.refresh_every)

    def featurize(self, tx: dict) -> tuple[dict[str, float], dict, float, list[tuple[str, int]]]:
        """Return (features, graph context, epoch, labels that matured at this instant). Read-only."""
        ts = to_epoch(tx["timestamp"])
        released = self.graph.release(ts)
        ctx = self.graph.context(tx)
        return self.features.compute(tx, ts, ctx), ctx, ts, released

    def commit(self, tx: dict, ts: float, label: int | None = None) -> None:
        """Fold the event into state. ``label`` (if its truth is known) matures after ``label_delay``."""
        self.features.observe(tx, ts)
        self.graph.observe(tx)
        if label is not None:
            self.graph.schedule_label(tx, label, ts + self.label_delay)


def compute_features(stream: list[dict], label_delay: float = 4 * 3600.0, refresh_every: int = 500,
                     state: StreamState | None = None) -> tuple[pd.DataFrame, StreamState]:
    """Walk a labelled stream once and return a feature frame plus the warmed-up online state."""
    state = state or StreamState(label_delay=label_delay, refresh_every=refresh_every)
    rows = []
    for tx in stream:
        features, _, ts, _ = state.featurize(tx)
        label = int(tx["is_fraud"])
        rows.append({"transaction_id": tx["transaction_id"], "ts": ts, "label": label,
                     "amount": tx["amount"], "scenario": tx.get("scenario", ""), **features})
        state.commit(tx, ts, label)
    frame = pd.DataFrame(rows)
    return frame[["transaction_id", "ts", "label", "amount", "scenario", *FEATURE_NAMES]], state


def warm_state(stream: list[dict], label_delay: float, refresh_every: int = 500) -> StreamState:
    """Rebuild online state from history (used at service start-up)."""
    return compute_features(stream, label_delay, refresh_every)[1]

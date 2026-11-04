from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from app.config import get_settings
from app.pipeline import compute_features
from app.synthetic import DriftSpec, make_transactions, strip_labels

DELAY = 4 * 3600.0


@pytest.fixture(scope="session")
def small_stream():
    return make_transactions(6000, seed=42)


@pytest.fixture(scope="session")
def small_frame(small_stream):
    frame, state = compute_features(small_stream, DELAY)
    return frame, state


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A real ServiceBundle trained on 6k rows with a delayed label feed for a 600-row drifting live stream."""
    from app.services import ServiceBundle
    tmp = tmp_path_factory.mktemp("fs")
    settings = replace(get_settings(), data_dir=tmp, train_rows=6000, auto_retrain=False, mlflow_uri=None,
                       label_delay_seconds=DELAY, threshold=None, drift_min_rows=150, drift_check_every=50,
                       retrain_min_labels=50)
    base = make_transactions(6000, seed=42)
    base_end = max(datetime.fromisoformat(t["timestamp"]) for t in base)
    live = make_transactions(600, seed=73, world_seed=42, start=base_end + timedelta(hours=1), id_prefix="live",
                             drift=DriftSpec(start_index=200, ramp_rows=150))
    feed = {t["transaction_id"]: t["is_fraud"] for t in live}
    bundle = ServiceBundle.create(settings, label_feed=feed)
    bundle.live_stream = live
    bundle.base_stream = base
    return bundle

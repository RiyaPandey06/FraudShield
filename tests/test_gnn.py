from __future__ import annotations

import numpy as np

from app.features import TABULAR_FEATURES
from experiments.gnn import MessagePassingNet, message_passing_features


def test_message_passing_is_causal_and_uses_shared_devices(small_stream, small_frame):
    frame, _ = small_frame
    tab = frame[TABULAR_FEATURES].to_numpy(float)
    full = message_passing_features(small_stream, tab)
    prefix = message_passing_features(small_stream[:1500], tab[:1500])
    np.testing.assert_allclose(full[:1500], prefix)          # later events never change earlier rows
    assert full[:, -1].max() > 0                              # some rows do have neighbours via shared devices


def test_message_passing_net_learns_a_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 6))
    y = (x[:, 0] + 0.5 * rng.normal(size=1500) > 1.2).astype(int)
    model = MessagePassingNet(0).fit(x, y)
    p = model.predict_proba(x)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2

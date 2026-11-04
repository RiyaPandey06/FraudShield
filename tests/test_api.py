"""API tests; skipped automatically when FastAPI is not installed."""
from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.synthetic import make_transactions, strip_labels  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("DATA_DIR", str(tmp_path_factory.mktemp("api")))
    monkeypatch_module.setenv("TRAIN_ROWS", "6000")
    monkeypatch_module.setenv("AUTO_RETRAIN", "false")
    from app.main import app
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    patch = MonkeyPatch()
    yield patch
    patch.undo()


def test_health_score_alert_feedback_flow(client):
    assert client.get("/api/health").json()["status"] == "ok"
    stream = [strip_labels(t) for t in make_transactions(400, seed=5, world_seed=42, id_prefix="api")]
    alert_id = None
    for tx in stream:
        body = client.post("/api/score", json=tx).json()
        assert {"risk_score", "decision", "reasons", "explanation", "latency_ms"} <= set(body)
        alert_id = alert_id or (tx["transaction_id"] if body["decision"] == "alert" else None)
    assert alert_id, "expected at least one alert in 400 transactions"
    assert client.get(f"/api/alerts/{alert_id}").json()["reasons"]
    fb = client.post(f"/api/alerts/{alert_id}/feedback", json={"label": "fraud", "review_ms": 3000, "arm": "raw"})
    assert fb.status_code == 200
    assert client.post("/api/alerts/nope/feedback", json={"label": "fraud"}).status_code == 404
    assert client.get("/api/summary").json()["feedback_count"] == 1
    assert "fraudshield_scoring_latency_seconds" in client.get("/metrics").text or client.get("/metrics").status_code == 200

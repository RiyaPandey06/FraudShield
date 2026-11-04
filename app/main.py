from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app import metrics
from app.config import get_settings
from app.schemas import FeedbackIn, ScoreResponse, TransactionIn
from app.services import ServiceBundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The API keeps its own warmed state for direct /api/score calls; the stream worker keeps another.
    app.state.bundle = ServiceBundle.create(get_settings())
    yield


app = FastAPI(title="FraudShield API", version="1.0.0", lifespan=lifespan,
              description="Streaming fraud scoring with graph context, grounded explanations, drift monitoring.")
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def bundle_for(request: Request) -> ServiceBundle:
    return request.app.state.bundle


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health(request: Request) -> dict:
    meta = bundle_for(request).model.metadata
    return {"status": "ok", "model_version": meta["model_version"], "backend": meta["backend"],
            "model_source": meta.get("model_source")}


@app.post("/api/score", response_model=ScoreResponse)
def score(transaction: TransactionIn, request: Request) -> dict:
    return bundle_for(request).process(transaction.model_dump(mode="json"))


@app.get("/api/alerts")
def alerts(request: Request, limit: int = 50, status: str = "open") -> list[dict]:
    return bundle_for(request).database.list_alerts(limit=max(1, min(limit, 200)), status=status)


@app.get("/api/alerts/{transaction_id}")
def alert_detail(transaction_id: str, request: Request) -> dict:
    alert = bundle_for(request).database.get_alert(transaction_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@app.post("/api/alerts/{transaction_id}/feedback")
def feedback(transaction_id: str, body: FeedbackIn, request: Request) -> dict:
    saved = bundle_for(request).database.record_feedback(transaction_id, body.label, body.analyst, body.note,
                                                         body.review_ms, body.arm)
    if not saved:
        raise HTTPException(status_code=404, detail="Alert not found")
    metrics.FEEDBACK.labels(body.label).inc()
    return {"transaction_id": transaction_id, "status": body.label,
            "message": "Feedback stored; it is used by the next retraining run."}


@app.get("/api/summary")
def summary(request: Request) -> dict:
    bundle = bundle_for(request)
    meta = bundle.model.metadata
    return {**bundle.database.summary(), "model_version": meta["model_version"], "backend": meta["backend"],
            "explainer": meta["explainer"], "threshold": bundle.threshold, "model_metrics": meta["metrics"],
            "drift": bundle.database.latest_drift(), "retrains": bundle.database.retrain_history(5)}


@app.get("/api/drift")
def drift(request: Request) -> dict:
    db = bundle_for(request).database
    return {"latest": db.latest_drift(), "history": db.drift_history()}


@app.get("/api/model")
def model_info(request: Request) -> dict:
    meta = dict(bundle_for(request).model.metadata)
    meta.pop("reference", None)
    meta.pop("ref_vector", None)
    return meta


@app.post("/api/retrain")
def retrain(request: Request) -> dict:
    return bundle_for(request).retrain("manual")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(metrics.generate_latest(), media_type=metrics.CONTENT_TYPE_LATEST)

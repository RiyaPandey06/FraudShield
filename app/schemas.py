from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TransactionIn(BaseModel):
    transaction_id: str = Field(min_length=3, max_length=80)
    timestamp: datetime
    account_id: str = Field(min_length=1, max_length=80)
    device_id: str = Field(min_length=1, max_length=80)
    merchant_id: str = Field(min_length=1, max_length=80)
    amount: float = Field(gt=0, le=1_000_000)
    country: str = Field(default="IN", min_length=2, max_length=3)
    transaction_type: str = Field(default="card_purchase", max_length=40)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: Literal["increases_risk", "reduces_risk"]
    label: str


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: Literal["alert", "allow"]
    threshold: float
    model_version: str
    latency_ms: float
    timing_ms: dict[str, float] = {}
    reasons: list[Reason]
    graph_context: dict
    explanation: str


class FeedbackIn(BaseModel):
    label: Literal["fraud", "legitimate"]
    analyst: str = Field(default="demo-analyst", max_length=80)
    note: str = Field(default="", max_length=1000)
    review_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    arm: Literal["raw", "explained"] | None = None

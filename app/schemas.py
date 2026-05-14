"""Pydantic request/response schemas."""
from typing import Literal
from pydantic import BaseModel, Field

from .config import MIN_TEXT_LEN, MAX_TEXT_LEN


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=MIN_TEXT_LEN, max_length=MAX_TEXT_LEN)


class PredictResponse(BaseModel):
    request_id: str
    label: Literal["POSITIVE", "NEGATIVE"]
    score: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class DriftSignal(BaseModel):
    name: str
    value: float
    threshold: float
    triggered: bool


class DriftResponse(BaseModel):
    overall_status: Literal["ok", "drift_detected", "insufficient_data"]
    window_size: int
    signals: list[DriftSignal]


class LogEvent(BaseModel):
    request_id: str
    ts: float
    text_hash: str
    text_len: int
    text_lang: str | None
    label: str
    score: float
    latency_ms: float
    model_version: str


class LogsResponse(BaseModel):
    events: list[LogEvent]

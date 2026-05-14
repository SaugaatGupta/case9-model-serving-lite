"""FastAPI app entrypoint."""
import uuid
from contextlib import asynccontextmanager
from time import perf_counter

import structlog
from fastapi import FastAPI, HTTPException
from langdetect import DetectorFactory, LangDetectException, detect

from .config import DRIFT_WINDOW, MODEL_VERSION, REFERENCE_STATS_PATH
from .drift import compute_drift, drift_state, load_reference, save_reference, build_reference
from .logging_config import configure_logging, get_logger
from .model import get_model
from .schemas import (
    DriftResponse,
    HealthResponse,
    LogsResponse,
    PredictRequest,
    PredictResponse,
)
from .store import hash_text, init_db, insert_event, recent_events, window_events


DetectorFactory.seed = 0

configure_logging()
log = get_logger("app.main")


def _ensure_reference() -> None:
    if REFERENCE_STATS_PATH.exists():
        return
    try:
        from datasets import load_dataset
        ds = load_dataset("sst2", split="validation")
        texts = [ds[i]["sentence"] for i in range(min(1000, len(ds)))]
        labels = ["POSITIVE" if ds[i]["label"] == 1 else "NEGATIVE"
                  for i in range(min(1000, len(ds)))]
        save_reference(build_reference(texts, labels))
        log.info("reference_stats_built", n=len(texts))
    except Exception as e:
        log.warning("reference_stats_build_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _ensure_reference()
    m = get_model()
    m.predict("warmup")  # dummy inference
    log.info("startup_complete", model_version=MODEL_VERSION)
    yield


app = FastAPI(title="Sentiment MLOps Service", version="1.0.0", lifespan=lifespan)


def _safe_detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    t0 = perf_counter()
    model = get_model()
    label, score, _ = model.predict(req.text)
    latency_ms = (perf_counter() - t0) * 1000.0
    lang = _safe_detect_lang(req.text)

    insert_event(
        request_id=request_id,
        text_hash=hash_text(req.text),
        text_len=len(req.text),
        text_lang=lang,
        label=label,
        score=score,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
    )
    drift_state.add(req.text)

    log.info(
        "prediction",
        label=label,
        score=score,
        latency_ms=latency_ms,
        text_len=len(req.text),
        text_lang=lang,
        model_version=MODEL_VERSION,
    )
    structlog.contextvars.clear_contextvars()
    return PredictResponse(
        request_id=request_id,
        label=label,
        score=score,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    try:
        model.predict("ping")
        loaded = True
    except Exception as e:
        log.error("health_inference_failed", error=str(e))
        loaded = False
    if not loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ok", model_loaded=True, model_version=MODEL_VERSION)


@app.get("/logs", response_model=LogsResponse)
def logs(limit: int = 20) -> LogsResponse:
    limit = max(1, min(limit, 1000))
    return LogsResponse(events=recent_events(limit=limit))


@app.get("/drift", response_model=DriftResponse)
def drift() -> DriftResponse:
    ref = load_reference()
    window = window_events(window=DRIFT_WINDOW)
    return compute_drift(window, ref or {})

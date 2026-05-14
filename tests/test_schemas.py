import pytest
from pydantic import ValidationError

from app.schemas import DriftSignal, HealthResponse, PredictRequest, PredictResponse


def test_predict_request_empty_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_request_too_long_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(text="a" * 2001)


def test_predict_request_valid():
    assert PredictRequest(text="hi").text == "hi"
    assert PredictRequest(text="a" * 2000).text.startswith("a")


def test_predict_response_label_constrained():
    r = PredictResponse(
        request_id="abc-123",
        label="POSITIVE",
        score=0.93,
        model_version="v1",
        latency_ms=12.4,
    )
    assert r.label == "POSITIVE"
    with pytest.raises(ValidationError):
        PredictResponse(
            request_id="abc-123",
            label="MAYBE",
            score=0.5,
            model_version="v1",
            latency_ms=1.0,
        )


def test_health_response_shape():
    h = HealthResponse(status="ok", model_loaded=True, model_version="v1")
    assert h.status == "ok"


def test_drift_signal_shape():
    s = DriftSignal(name="x", value=0.1, threshold=0.05, triggered=True)
    assert s.triggered is True

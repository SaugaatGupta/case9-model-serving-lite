from fastapi.testclient import TestClient

from app.main import app


def test_predict_happy_path():
    with TestClient(app) as c:
        r = c.post("/predict", json={"text": "I absolutely love this!"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in ("POSITIVE", "NEGATIVE")
        assert 0.0 <= body["score"] <= 1.0
        assert isinstance(body["request_id"], str) and len(body["request_id"]) > 10
        assert "latency_ms" in body and body["latency_ms"] >= 0
        assert "model_version" in body


def test_predict_empty_returns_422():
    with TestClient(app) as c:
        r = c.post("/predict", json={"text": ""})
        assert r.status_code == 422


def test_predict_too_long_returns_422():
    with TestClient(app) as c:
        r = c.post("/predict", json={"text": "x" * 2001})
        assert r.status_code == 422


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

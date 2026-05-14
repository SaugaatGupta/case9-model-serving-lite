---
title: Case 9 Model Serving Lite
emoji: 🧪
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# case9-model-serving-lite

**Live:** `https://huggingface.co/spaces/<YOUR_USERNAME>/case9-model-serving-lite`

A small but production-shaped sentiment-analysis service: FastAPI + DistilBERT-SST2, structured JSON logs, SQLite event store, four drift signals, and a PR-gated retraining workflow.

## Quickstart

```bash
# local
pip install -r requirements.txt
python scripts/build_reference.py            # one-shot, builds models/reference_stats.json
uvicorn app.main:app --host 0.0.0.0 --port 7860

curl -X POST localhost:7860/predict \
  -H 'content-type: application/json' \
  -d '{"text":"I love this"}'

# docker
docker build -t case9 .
docker run -p 7860:7860 case9
```

## Endpoints

- `POST /predict` — `{text}` → `{request_id, label, score, model_version, latency_ms}`
- `GET  /health`  — confirms the model is loaded and runs a dummy inference
- `GET  /logs?limit=20` — recent events from the SQLite store
- `GET  /drift`  — four drift signals + `overall_status`
- `GET  /docs`   — Swagger UI

## Stack

| Layer | Choice |
| --- | --- |
| Serving | FastAPI + Uvicorn |
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Validation | Pydantic v2 (`min_length=1`, `max_length=2000`) |
| Logging | structlog (JSON to stdout, request_id-tagged) |
| Event store | SQLite at `/tmp/events.db`, sha256 hashes only |
| Lang detect | langdetect |
| Drift | scipy KS-test, vocabulary Jaccard, ratio shifts |
| Retrain head | DistilBERT mean-pool + sklearn `LogisticRegression` |
| CI | GitHub Actions — ruff + pytest |
| Container | `python:3.11-slim`, multi-stage, non-root, model baked in |
| Deploy | Hugging Face Spaces (Docker SDK, port 7860) |

## Drift demo

```bash
# start the service, then in another shell:
python scripts/simulate_drift.py --url http://localhost:7860 --n 500
curl localhost:7860/drift  # → overall_status: drift_detected
```

## Retraining

PRs that touch `data/train.csv` trigger `.github/workflows/retrain.yml`:
1. Re-train `models/candidate.pkl` (DistilBERT embeddings → LogReg).
2. Evaluate against `eval/held_out.csv` (frozen).
3. Fail if `baseline.f1 − candidate.f1 > 0.02`.
4. Post a before/after metrics comment on the PR.

## Project structure

```
app/        FastAPI service (main, model, drift, store, schemas, config)
scripts/    build_reference, retrain, evaluate, simulate_drift
tests/      predict, drift, schemas
data/       train.csv (200 rows)
eval/       held_out.csv (200 rows, FROZEN)
models/     baseline.json (+ reference_stats.json built at startup)
.github/    ci.yml, retrain.yml
```

See [MONITORING.md](MONITORING.md) and [DECISIONS.md](DECISIONS.md) for the writeups.

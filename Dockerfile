# syntax=docker/dockerfile:1.6

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download model weights at build time so cold starts don't fail.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
n='distilbert-base-uncased-finetuned-sst-2-english'; \
AutoTokenizer.from_pretrained(n); AutoModelForSequenceClassification.from_pretrained(n)"


# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    PORT=7860

RUN useradd -m -u 1000 appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/hf_cache /opt/hf_cache

WORKDIR /app
COPY --chown=appuser:appuser . /app

RUN mkdir -p /tmp && chown -R appuser:appuser /opt/hf_cache /app /tmp
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

# Decisions, assumptions, and trade-offs

## Five assumptions
1. **English-only traffic.** The base model is English; non-English is treated as drift, not silently handled.
2. **CPU-only serving is acceptable.** HF Spaces free tier is CPU; DistilBERT fits comfortably with ~150–300 ms p50 on short inputs.
3. **Privacy beats convenience.** Raw text is never persisted — only a sha256 hash, length, and detected language. This keeps the event store useful for drift while staying defensible if leaked.
4. **A single replica is fine for this case.** No horizontal scaling or shared state required; SQLite at `/tmp/events.db` is per-pod and that's intentional.
5. **The retraining workflow is a quality gate, not an autopromoter.** It runs on PRs that touch training data, fails on regression, and posts a comment — a human still merges.

## Trade-offs

| Choice | Picked | Rejected | Why |
| --- | --- | --- | --- |
| Event store | SQLite at `/tmp` | Postgres / S3 + Athena | Zero ops, fast enough for 500-event windows, ephemeral matches Space lifetime. |
| Drift signals | 4 lightweight statistics | Evidently / WhyLabs / Fiddler | This is one service; managed drift tools add dependency surface that isn't justified yet. |
| Retraining head | DistilBERT embeddings + LogReg | Full fine-tune | Fits in <2 min on a GitHub runner; full fine-tune blows the budget and adds GPU coupling. |
| Logging | structlog JSON to stdout | Centralised aggregator | Spaces collects stdout; aggregator would be the next step, not the first. |
| Language detection | langdetect | fastText / lid176 | Pure Python, no extra weights to ship; accuracy is sufficient for a ratio. |
| Container | python:3.11-slim multi-stage | distroless / nvidia base | Smallest reliable footprint for CPU PyTorch; non-root user added. |
| Reference stats | 1000 SST-2 dev samples | Synthetic / customer data | SST-2 is what the model was trained on; matches the regime we expect at launch. |

## De-scoped (intentionally)
- **Shadow deployment.** Mirror traffic to a candidate model and compare offline. Worth the build once we have two model versions worth comparing; today we have one.
- **Feature store.** Not needed for a stateless text-in/label-out endpoint. Would be on the table the moment features become non-trivial (user history, account flags, etc.).
- **Online A/B test infrastructure.** Adjacent to shadow deploys; same justification for deferring.
- **Model registry beyond a pickled head.** The retrain workflow produces `models/candidate.pkl`; promotion is manual.
- **Per-user rate limiting and auth.** Belongs at the edge, not in the service.

## AI assistance disclosure
This repository was scaffolded with help from Claude (Anthropic). Concretely: the file layout, Dockerfile, drift module, retrain/evaluate scripts, GitHub Actions workflows, and writeups were generated and then reviewed and edited by the author. Pinned dependency versions, thresholds in `config.py`, and the structure of the tests were chosen and verified manually. No proprietary code or data was shared with the model.

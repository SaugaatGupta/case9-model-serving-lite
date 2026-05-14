# Monitoring: How would I know this model is failing in production before customers do?

## Symptoms vs causes
Customers report **symptoms** ("the labels feel off", "it's slow", "it crashed").
We monitor **causes** so we see them first: latency, errors, distributional shifts in input and output, and confidence collapse. The job of monitoring is to make the cause observable before the symptom is felt — and to make the right cause obvious so the on-call doesn't chase ghosts.

## What we watch

**System health.** Latency p50 / p95 / p99 on `/predict`. Error rate (4xx separated from 5xx — 4xx is usually a client). Throughput (requests/min). Process resident memory and CPU. Container restarts.

**Input drift.** Distribution of input length (KS-test vs the SST-2 reference quantiles). Non-English ratio (langdetect — anything other than `en`). Vocabulary overlap (Jaccard of last-500 tokens vs reference top-100). All three are surfaced by `GET /drift`.

**Output drift.** Predicted-label positive/negative ratio vs reference. Confidence distribution: a sudden collapse to ~0.5 means the model has effectively lost signal; a sudden spike to ~1.0 on everything means it has learned to print one label. Both matter.

**Data integrity.** Rate of 422s (validation failures) — if it spikes, a client just shipped a bad release. Rate of `unknown` language — usually empty/garbled input.

## Alert thresholds and rationale

| Signal | Threshold | Why |
| --- | --- | --- |
| p95 latency | > 800 ms for 5 min | DistilBERT on CPU sits ~150–300 ms; 800 ms means a queueing or memory issue. |
| 5xx rate | > 1% over 5 min | Anything above background is a real bug. |
| 422 rate | > 5% over 15 min | Client contract drift — page the integration owner. |
| Length KS p-value | < 0.01 over 500 events | Inputs no longer look like training data. |
| Non-English ratio | > 10% over 500 events | Model is English-only; foreign traffic = silent quality loss. |
| Vocab Jaccard | < 0.30 over 500 events | The world the model knows has changed. |
| Label positive shift | > 0.20 vs reference | Either the world changed, or the model is broken. Investigate before assuming the world. |
| Confidence median | < 0.6 over 1h | Model is hedging — something is off-distribution. |

The thresholds in `app/config.py` mirror these (`DRIFT_LENGTH_KS_P`, `DRIFT_NON_ENGLISH_RATIO`, `DRIFT_JACCARD_MIN`, `DRIFT_LABEL_DELTA`).

## Investigation playbook
When an alert fires, run this in order. Latency or 5xx → check the deploy timeline and resource graphs first (almost always a regression or a memory leak). 422 spike → diff recent client releases. Drift alerts → look at `GET /logs?limit=200` and at the simulated examples; check whether one client is responsible (group by ip/user-agent at the edge), then decide whether to retrain, route, or do nothing. Confidence collapse with no input drift → suspect the model artifact itself; verify the version banner in `/health` matches what was deployed.

## What monitoring does NOT catch
Monitoring catches the model failing the data it sees. It does not catch the model succeeding on the wrong objective — e.g. confidently predicting POSITIVE on sarcasm that humans would call negative. That only surfaces through periodic labeled audits, customer feedback, and shadow traffic against a stronger reference model. Monitoring also doesn't catch slow upstream poisoning where the input distribution drifts within thresholds for weeks and only then crosses them — for that, plot rolling 30-day trends alongside the 5-minute alerts.

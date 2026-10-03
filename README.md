# ML Monitoring and Drift Detection

A monitoring service that treats model health as several different questions:

- Are the **features** still from the same distribution? (data drift, PSI)
- Are the **scores** still from the same distribution? (prediction drift)
- When labels exist, is **accuracy** still above a floor? (performance / concept-adjacent)
- Is the **API** erroring or slowing down? (system)

Alerts carry a **severity** and an **action**. The action `retrain_signal` is a flag on `/v1/report`, not an automatic training job. That split is deliberate: monitoring should not silently mutate Production.

```
prediction events (features, score, optional label, latency, error)
        → jsonl window
        → compare to reference snapshot
        → report + Prometheus text
        → alerts: watch | ticket | page
        → retrain_signal true/false
```

## Local setup

```bash
pip install -e ".[dev]"
uvicorn mlmonitor.serving:app --port 8000
```

POST events to `/v1/events`, read `/v1/report`.

## After an alert

| Alert | Action encoded in the payload |
| --- | --- |
| data_drift ticket | `retrain_signal` — freeze promotions, run a new training window |
| prediction_drift | compare score histograms before touching the model |
| concept_or_performance page | `retrain_signal` if labels confirm a drop |
| error_rate page | page on-call; this is serving, not necessarily the model |

## Tests

```bash
pytest -q
```

## Trade-offs

- Reference snapshot is generated at first boot for local use. A production deployment should freeze the training matrix instead.
- Concept drift is only approximated when labels arrive. Without labels the service will not pretend to know.

## What I would improve next

- Export gauges from a sidecar on a schedule rather than computing on scrape.
- Multi-feature dashboards as Grafana JSON generated from the report schema.

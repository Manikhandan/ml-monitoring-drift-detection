# Architecture

EventStore is an append-only jsonl log plus a reference snapshot.
Monitor.evaluate() is a pure window computation so tests do not need Prometheus.
The API is a thin ingest/report surface.

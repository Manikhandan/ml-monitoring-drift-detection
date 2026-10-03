from __future__ import annotations

import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlmonitor.config import Settings, get_settings
from mlmonitor.monitor import Monitor
from mlmonitor.store import EventStore


class EventIn(BaseModel):
    features: list[float] = Field(..., min_length=1)
    score: float
    label: int | None = None
    error: bool = False
    latency_ms: float = 0.0
    quality: str | None = None


def _seed_reference(store: EventStore, rng: np.random.Generator) -> None:
    if store.ref.exists():
        return
    features = rng.normal(0, 1, size=(400, 5)).tolist()
    scores = rng.uniform(0.2, 0.8, size=400).tolist()
    store.write_reference({"features": features, "scores": scores})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = EventStore(settings.store_dir)
    rng = np.random.default_rng(0)
    _seed_reference(store, rng)
    monitor = Monitor(store, settings)
    app = FastAPI(title="mlmonitor")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": store.ref.exists()}

    @app.post("/v1/events")
    def ingest(body: EventIn):
        store.append(body.model_dump())
        return {"accepted": True}

    @app.get("/v1/report")
    def report():
        return monitor.evaluate()

    @app.get("/metrics")
    def metrics():
        report = monitor.evaluate()
        lines = [
            f"ml_monitor_events {report['n']}",
            f"ml_monitor_error_rate {report['error_rate']}",
            f"ml_monitor_data_psi {report['data_psi']}",
            f"ml_monitor_prediction_psi {report['prediction_psi']}",
            f"ml_monitor_retrain_signal {int(report['retrain_signal'])}",
        ]
        return "\n".join(lines) + "\n"

    @app.post("/v1/alerts/ack")
    def ack(name: str):
        if not name:
            raise HTTPException(400, "name required")
        return {"acked": name, "id": str(uuid.uuid4())}

    return app


app = create_app()

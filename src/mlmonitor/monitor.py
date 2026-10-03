from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from mlmonitor.config import Settings
from mlmonitor.stats import ks_stat, mean_or_zero, psi
from mlmonitor.store import EventStore

Severity = Literal["info", "watch", "ticket", "page"]


@dataclass
class Alert:
    name: str
    severity: Severity
    value: float
    threshold: float
    action: str


class Monitor:
    def __init__(self, store: EventStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def evaluate(self) -> dict[str, Any]:
        events = self.store.read()[-self.settings.window :]
        reference = self.store.read_reference()
        features_ref = np.asarray(reference["features"], dtype=float)
        scores_ref = np.asarray(reference["scores"], dtype=float)
        features_cur = np.asarray([e["features"] for e in events], dtype=float) if events else np.empty((0, 1))
        scores_cur = np.asarray([e["score"] for e in events], dtype=float) if events else np.empty(0)
        labels = [e.get("label") for e in events if e.get("label") is not None]
        preds = [int(e["score"] >= 0.5) for e in events if e.get("label") is not None]
        errors = [e.get("error", False) for e in events]
        latencies = [e.get("latency_ms", 0.0) for e in events]

        data_psi = 0.0
        pred_psi = 0.0
        if len(features_cur) >= 20:
            cols = min(features_ref.shape[1], features_cur.shape[1])
            data_psi = mean_or_zero(
                [psi(features_ref[:, i], features_cur[:, i]) for i in range(cols)]
            )
            pred_psi = psi(scores_ref, scores_cur)
        accuracy = None
        if labels:
            accuracy = sum(int(a == b) for a, b in zip(preds, labels, strict=False)) / len(labels)
        error_rate = sum(1 for item in errors if item) / len(events) if events else 0.0
        quality_nulls = sum(1 for e in events if e.get("quality") == "null") / len(events) if events else 0.0

        alerts: list[Alert] = []
        if data_psi >= self.settings.psi_alert:
            alerts.append(Alert("data_drift", "ticket", data_psi, self.settings.psi_alert, "retrain_signal"))
        elif data_psi >= self.settings.psi_watch:
            alerts.append(Alert("data_drift", "watch", data_psi, self.settings.psi_watch, "review_window"))
        if pred_psi >= self.settings.psi_alert:
            alerts.append(Alert("prediction_drift", "ticket", pred_psi, self.settings.psi_alert, "compare_score_histogram"))
        if accuracy is not None and accuracy < self.settings.accuracy_floor:
            alerts.append(Alert("concept_or_performance", "page", accuracy, self.settings.accuracy_floor, "retrain_signal"))
        if error_rate >= self.settings.error_rate_alert:
            alerts.append(Alert("error_rate", "page", error_rate, self.settings.error_rate_alert, "page_oncall"))

        retrain = any(alert.action == "retrain_signal" for alert in alerts)
        return {
            "n": len(events),
            "throughput": len(events),
            "error_rate": error_rate,
            "latency_ms_mean": mean_or_zero(latencies),
            "data_psi": data_psi,
            "prediction_psi": pred_psi,
            "score_ks": float(ks_stat(scores_ref, scores_cur)) if len(scores_cur) >= 10 else 0.0,
            "accuracy": accuracy,
            "null_rate": quality_nulls,
            "alerts": [alert.__dict__ for alert in alerts],
            "retrain_signal": retrain,
        }

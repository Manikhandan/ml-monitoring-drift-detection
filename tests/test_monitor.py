from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from mlmonitor.config import Settings
from mlmonitor.serving import create_app
from mlmonitor.stats import psi


def test_psi_shift():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 300), rng.normal(3, 1, 300)) > 0.2


def test_report_and_retrain_signal(tmp_path: Path):
    settings = Settings(store_dir=tmp_path, window=250, psi_alert=0.2, accuracy_floor=0.9)
    client = TestClient(create_app(settings))
    rng = np.random.default_rng(2)
    for _ in range(80):
        features = rng.normal(4, 1, size=5).tolist()
        client.post(
            "/v1/events",
            json={
                "features": features,
                "score": 0.9,
                "label": 0,
                "error": False,
                "latency_ms": 12.0,
            },
        )
    report = client.get("/v1/report").json()
    assert report["n"] == 80
    assert "alerts" in report
    assert report["retrain_signal"] is True
    assert client.get("/health").json()["status"] == "ok"
    metrics = client.get("/metrics").text
    assert "ml_monitor_data_psi" in metrics


def test_error_rate_pages(tmp_path: Path):
    settings = Settings(store_dir=tmp_path / "err", window=50, error_rate_alert=0.2)
    client = TestClient(create_app(settings))
    for i in range(20):
        client.post(
            "/v1/events",
            json={
                "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                "score": 0.4,
                "error": i < 8,
                "latency_ms": 5.0,
            },
        )
    report = client.get("/v1/report").json()
    names = [alert["name"] for alert in report["alerts"]]
    assert "error_rate" in names
    assert client.post("/v1/events", json={"features": [], "score": 0.1}).status_code == 422


def test_insufficient_window_has_no_psi_alert(tmp_path: Path):
    settings = Settings(store_dir=tmp_path / "small", window=200, psi_alert=0.2)
    client = TestClient(create_app(settings))
    client.post(
        "/v1/events",
        json={"features": [0.1, 0.2, 0.3, 0.4, 0.5], "score": 0.4, "latency_ms": 3},
    )
    report = client.get("/v1/report").json()
    assert report["n"] == 1
    assert report["data_psi"] == 0.0
    assert report["accuracy"] is None
    assert report["retrain_signal"] is False


def test_accuracy_below_floor_without_errors(tmp_path: Path):
    settings = Settings(
        store_dir=tmp_path / "acc",
        window=50,
        accuracy_floor=0.8,
        error_rate_alert=0.9,
        psi_alert=9.0,
    )
    client = TestClient(create_app(settings))
    for _ in range(25):
        client.post(
            "/v1/events",
            json={
                "features": [0.0, 0.0, 0.0, 0.0, 0.0],
                "score": 0.9,
                "label": 0,
                "error": False,
                "latency_ms": 4,
            },
        )
    report = client.get("/v1/report").json()
    names = [alert["name"] for alert in report["alerts"]]
    assert "concept_or_performance" in names
    assert report["retrain_signal"] is True


def test_no_events_and_missing_labels(tmp_path: Path):
    settings = Settings(store_dir=tmp_path / "empty", window=50, psi_alert=0.2)
    client = TestClient(create_app(settings))
    empty = client.get("/v1/report").json()
    assert empty["n"] == 0
    assert empty["accuracy"] is None
    assert empty["retrain_signal"] is False
    client.post(
        "/v1/events",
        json={"features": [0.1, 0.2, 0.3, 0.4, 0.5], "score": 0.4, "latency_ms": 2},
    )
    unlabeled = client.get("/v1/report").json()
    assert unlabeled["accuracy"] is None


def test_error_rate_at_threshold(tmp_path: Path):
    settings = Settings(store_dir=tmp_path / "thr", window=50, error_rate_alert=0.5, psi_alert=9.0)
    client = TestClient(create_app(settings))
    for i in range(10):
        client.post(
            "/v1/events",
            json={
                "features": [0.1, 0.2, 0.3, 0.4, 0.5],
                "score": 0.4,
                "error": i < 5,
                "latency_ms": 4.0,
            },
        )
    report = client.get("/v1/report").json()
    names = [alert["name"] for alert in report["alerts"]]
    assert "error_rate" in names
    assert report["error_rate"] == 0.5

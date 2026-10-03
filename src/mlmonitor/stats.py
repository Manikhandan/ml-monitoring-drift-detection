from __future__ import annotations

import math

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    if len(reference) < 20 or len(current) < 20:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_h, _ = np.histogram(reference, bins=edges)
    cur_h, _ = np.histogram(current, bins=edges)
    ref_p = np.clip(ref_h / max(ref_h.sum(), 1), 1e-4, 1)
    cur_p = np.clip(cur_h / max(cur_h.sum(), 1), 1e-4, 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_stat(reference: np.ndarray, current: np.ndarray) -> float:
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    all_v = np.concatenate([ref_sorted, cur_sorted])
    ref_cdf = np.searchsorted(ref_sorted, all_v, side="right") / len(ref_sorted)
    cur_cdf = np.searchsorted(cur_sorted, all_v, side="right") / len(cur_sorted)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0

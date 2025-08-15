"""Benchmark utilities for measuring inference latency."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Iterable, Optional, Sequence

from wifi_activity_recognition.hardware.base import CSIData


def measure_latency(
    predictor: Callable[[CSIData], object],
    data: Iterable[CSIData],
    runs: int = 100,
    device: Optional[str] = None,
    warmup: int = 10,
) -> dict[str, float]:
    """Profile latency statistics for a predictor.

    Args:
        predictor: Callable that processes ``CSIData`` and performs inference.
        data: Iterable of packets used for benchmarking.
        runs: Number of repetitions used for averaging.
        device: Optional device to move ``predictor`` to if it exposes ``.to``.
        warmup: Number of warm-up iterations before measurement.

    Returns:
        Dictionary with ``mean_ms``, ``max_ms`` and ``min_ms`` latency values.
    """
    packets = list(data)
    if not packets:
        return {"mean_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}

    mover = getattr(predictor, "to", None)
    if device is not None and callable(mover):
        mover(device)

    # Warm-up runs to stabilise caches (important for edge devices)
    for _ in range(warmup):
        for packet in packets:
            predictor(packet)

    timings: list[float] = []
    for _ in range(runs):
        for packet in packets:
            start = time.perf_counter()
            predictor(packet)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

    return {
        "mean_ms": statistics.fmean(timings),
        "max_ms": max(timings),
        "min_ms": min(timings),
    }

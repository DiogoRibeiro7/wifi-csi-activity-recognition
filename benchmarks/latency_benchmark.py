"""Benchmark utilities for measuring inference latency."""

from __future__ import annotations

import time
from typing import Callable, Iterable

from wifi_activity_recognition.hardware.base import CSIData


def measure_latency(
    predictor: Callable[[CSIData], object],
    data: Iterable[CSIData],
    runs: int = 100,
) -> float:
    """Measure average latency for predictor inference.

    Args:
        predictor: Callable that processes ``CSIData`` and performs inference.
        data: Iterable of ``CSIData`` packets used for benchmarking.
        runs: Number of times to repeat inference for averaging.

    Returns:
        Average latency in milliseconds.
    """
    packets = list(data)
    if not packets:
        return 0.0

    start = time.perf_counter()
    for _ in range(runs):
        for packet in packets:
            predictor(packet)
    end = time.perf_counter()
    total_ops = runs * len(packets)
    return (end - start) * 1000 / total_ops

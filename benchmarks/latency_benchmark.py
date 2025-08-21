"""Benchmark utilities for measuring inference latency."""

from __future__ import annotations

import statistics
import time
from typing import Callable, Iterable, MutableMapping, Optional, Sequence, Union

import numpy as np

from wifi_activity_recognition.hardware.base import CSIData


def measure_latency(
    predictor: Callable[[CSIData], object],
    data: Iterable[CSIData],
    runs: int = 100,
    device: Optional[str] = None,
    warmup: int = 10,
    devices: Optional[Sequence[str]] = None,
    percentiles: Sequence[int] = (50, 95, 99),
) -> Union[MutableMapping[str, float], MutableMapping[str, MutableMapping[str, float]]]:
    """Profile latency statistics for a predictor.

    Args:
        predictor: Callable that processes ``CSIData`` and performs inference.
        data: Iterable of packets used for benchmarking.
        runs: Number of repetitions used for averaging.
        device: Optional device to move ``predictor`` to if it exposes ``.to``.
        warmup: Number of warm-up iterations before measurement.
        devices: Optional sequence of devices for cross-platform testing.
        percentiles: Percentile values to compute (e.g. ``(50, 95, 99)``).

    Returns:
        Dictionary with latency statistics in milliseconds. If ``devices`` is
        supplied, a nested mapping of ``device -> metrics`` is returned.
    """
    if devices is not None:
        results: MutableMapping[str, MutableMapping[str, float]] = {}
        for dev in devices:
            results[dev] = measure_latency(
                predictor,
                data,
                runs=runs,
                device=dev,
                warmup=warmup,
                percentiles=percentiles,
            )  # type: ignore[assignment]
        return results

    packets = list(data)
    if not packets:
        base = {"mean_ms": 0.0, "max_ms": 0.0, "min_ms": 0.0}
        for p in percentiles:
            base[f"p{p}_ms"] = 0.0
        return base

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

    stats = {
        "mean_ms": statistics.fmean(timings),
        "max_ms": max(timings),
        "min_ms": min(timings),
    }
    pct_values = np.percentile(timings, percentiles)
    for p, val in zip(percentiles, pct_values):
        stats[f"p{p}_ms"] = float(val)
    return stats

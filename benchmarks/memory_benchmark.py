"""Benchmark utilities for profiling memory usage during streaming."""

from __future__ import annotations

import gc
import tracemalloc
from typing import Callable, Iterable

from wifi_activity_recognition.hardware.base import CSIData


def measure_memory_usage(
    func: Callable[[Iterable[CSIData]], object],
    data: Iterable[CSIData],
    *,
    optimize: bool = False,
) -> int:
    """Measure peak memory usage in bytes of ``func`` over ``data``.

    Args:
        func: Function processing an iterable of ``CSIData``.
        data: Iterable of CSI packets passed to ``func``.
        optimize: If ``True`` run ``gc.collect`` before measuring to minimise
            background noise in the results.

    Returns:
        Peak memory in bytes observed during function execution.
    """
    if optimize:
        gc.collect()
    tracemalloc.start()
    func(data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def detect_memory_leak(
    func: Callable[[Iterable[CSIData]], object],
    data: Iterable[CSIData],
    runs: int = 10,
    threshold: int = 1024,
) -> bool:
    """Detect potential memory leaks by repeatedly executing ``func``.

    The function runs ``func`` ``runs`` times and reports ``True`` if the
    remaining allocated memory after execution exceeds ``threshold`` bytes.

    Args:
        func: Function consuming an iterable of ``CSIData``.
        data: Iterable of CSI packets to feed into ``func``.
        runs: Number of times to execute ``func``.
        threshold: Memory threshold in bytes signalling a leak.

    Returns:
        ``True`` if a potential leak is detected, ``False`` otherwise.
    """
    tracemalloc.start()
    for _ in range(runs):
        func(data)
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current > threshold

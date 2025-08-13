"""Benchmark utilities for profiling memory usage during streaming."""

from __future__ import annotations

import tracemalloc
from typing import Callable, Iterable

from wifi_activity_recognition.hardware.base import CSIData


def measure_memory_usage(
    func: Callable[[Iterable[CSIData]], object],
    data: Iterable[CSIData],
) -> int:
    """Measure peak memory usage in bytes of ``func`` over ``data``.

    Args:
        func: Function processing an iterable of ``CSIData``.
        data: Iterable of CSI packets passed to ``func``.

    Returns:
        Peak memory in bytes observed during function execution.
    """
    tracemalloc.start()
    func(data)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak

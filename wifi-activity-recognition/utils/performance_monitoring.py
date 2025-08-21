"""Real-time performance monitoring utilities."""

from __future__ import annotations

import resource
import time
from typing import List

import numpy as np


class PerformanceMonitor:
    """Track latency and memory usage for streaming pipelines."""

    def __init__(self) -> None:
        """Initialize empty statistics and counters."""
        self.latencies: List[float] = []
        self.start_time = time.perf_counter()
        self.processed = 0
        self.dropped = 0

    # ------------------------------------------------------------------
    def record_latency(self, duration_ms: float) -> None:
        """Record an inference latency measurement in milliseconds."""
        self.latencies.append(duration_ms)

    def record_processed(self) -> None:
        """Increment processed packet counter."""
        self.processed += 1

    def record_dropped(self) -> None:
        """Increment dropped packet counter."""
        self.dropped += 1

    def latency_percentile(self, percentile: float = 0.95) -> float:
        """Return latency percentile in milliseconds."""
        if not self.latencies:
            return 0.0
        data = np.sort(np.array(self.latencies))
        idx = int(len(data) * percentile)
        idx = min(max(idx, 0), len(data) - 1)
        return float(data[idx])

    # ------------------------------------------------------------------
    def memory_mb(self) -> float:
        """Return memory usage of the process in megabytes."""
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage_kb / 1024.0

    def packet_rate(self) -> float:
        """Return processed packets per second."""
        elapsed = time.perf_counter() - self.start_time
        return self.processed / elapsed if elapsed > 0 else 0.0

    def drop_rate(self) -> float:
        """Return fraction of packets dropped."""
        total = self.processed + self.dropped
        return self.dropped / total if total else 0.0

    def reset(self) -> None:
        """Clear collected statistics."""
        self.latencies.clear()
        self.start_time = time.perf_counter()
        self.processed = 0
        self.dropped = 0

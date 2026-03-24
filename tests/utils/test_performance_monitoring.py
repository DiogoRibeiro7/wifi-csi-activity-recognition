"""Tests for performance monitoring utilities."""

from pathlib import Path

from wifi_activity_recognition.utils import (  # type: ignore  # noqa: E402
    PerformanceMonitor,
)


def test_performance_monitor_records_and_resets() -> None:
    """Monitor tracks and clears latency statistics."""
    mon = PerformanceMonitor()
    mon.record_latency(10.0)
    mon.record_latency(20.0)
    mon.record_processed()
    mon.record_dropped()
    assert mon.latency_percentile(0.5) >= 10.0
    assert mon.memory_mb() >= 0.0
    assert mon.packet_rate() >= 0.0
    assert mon.drop_rate() == 0.5
    mon.reset()
    assert mon.latencies == []
    assert mon.processed == 0
    assert mon.dropped == 0



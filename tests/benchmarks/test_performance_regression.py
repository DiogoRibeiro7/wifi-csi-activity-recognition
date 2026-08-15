"""Automated performance regression detection.

Wall-clock thresholds are unreliable on shared CI runners: a green build on a
quiet runner and a red one on a noisy runner say nothing about the code. These
checks are therefore built from properties that do not depend on machine speed:

* **complexity** -- how cost grows with input size, which is a property of the
  algorithm rather than the host
* **memory** -- ``tracemalloc`` counts Python allocations, which is stable
  across machines for a fixed workload
* **relative** -- two measurements taken on the same host in the same run
* **invariants** -- orderings that must hold for any correct measurement

Absolute time ceilings appear only as deliberately generous backstops, marked
``slow`` so they can be excluded.

See ``docs/performance_policy.md`` for the policy these enforce.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, List

import numpy as np
import pytest

from benchmarks.latency_benchmark import measure_latency
from benchmarks.memory_benchmark import (
    detect_memory_leak,
    measure_memory_usage,
    profile_memory_usage,
)
from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.preprocessing import segment_windows

# Growing the input 4x costs ~4x if linear and ~16x if quadratic. Failing above
# 8x sits between the two in log space: wide enough to absorb runner noise,
# tight enough that quadratic blow-up cannot slip through.
GROWTH_FACTOR = 4
SUPERLINEAR_BUDGET = 8.0


def _packet(subcarriers: int = 30) -> CSIData:
    amplitude = np.random.rand(1, 1, subcarriers).astype(np.float32)
    return CSIData(
        timestamp=time.time(),
        amplitude=amplitude,
        phase=np.zeros_like(amplitude),
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=subcarriers,
    )


def _best_of(operation: Callable[[], object], repeats: int = 5) -> float:
    """Return the fastest of several runs, in seconds.

    The minimum is far more stable than the mean under contention: noise can
    only ever make a run slower, so the fastest observation is the closest
    estimate of the true cost.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        best = min(best, time.perf_counter() - start)
    return best


def _scaling_ratio(work: Callable[[int], object], base_size: int) -> float:
    """Cost ratio when the input grows by ``GROWTH_FACTOR``."""
    small = _best_of(lambda: work(base_size))
    large = _best_of(lambda: work(base_size * GROWTH_FACTOR))
    # Guard against a clock too coarse to measure the small case.
    return large / max(small, 1e-6)


# ---------------------------------------------------------------------------
# Complexity: cost must not grow super-linearly in the hot paths
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_segmentation_scales_linearly_with_sequence_length() -> None:
    """Windowing a stream must stay linear in the number of packets.

    Segmentation builds overlapping windows by slicing; a change that copies
    the whole sequence per window turns this quadratic, which stays invisible
    to correctness tests.
    """
    packets = [_packet() for _ in range(64 * GROWTH_FACTOR)]

    def work(size: int) -> object:
        return segment_windows(packets[:size], window_size=8, overlap=0.5)

    ratio = _scaling_ratio(work, base_size=64)
    assert ratio < SUPERLINEAR_BUDGET, (
        f"segmentation cost grew {ratio:.1f}x for a {GROWTH_FACTOR}x larger "
        f"input; expected roughly {GROWTH_FACTOR}x"
    )


@pytest.mark.regression
def test_packet_construction_scales_linearly_with_subcarriers() -> None:
    """CSIData validation must stay linear in array size."""

    def work(subcarriers: int) -> object:
        return [_packet(subcarriers) for _ in range(20)]

    ratio = _scaling_ratio(work, base_size=64)
    assert (
        ratio < SUPERLINEAR_BUDGET
    ), f"packet construction grew {ratio:.1f}x for {GROWTH_FACTOR}x the data"


# ---------------------------------------------------------------------------
# Memory: bounded per unit of work, and no accumulation across runs
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_streaming_memory_scales_with_the_stream_not_faster() -> None:
    """Peak memory must grow roughly in proportion to the data consumed."""

    def consume(data: Iterable[CSIData]) -> List[CSIData]:
        return list(data)

    small = measure_memory_usage(consume, [_packet() for _ in range(50)])
    large = measure_memory_usage(
        consume, [_packet() for _ in range(50 * GROWTH_FACTOR)]
    )

    ratio = large / max(small, 1)
    assert (
        ratio < SUPERLINEAR_BUDGET
    ), f"peak memory grew {ratio:.1f}x for a {GROWTH_FACTOR}x longer stream"


@pytest.mark.regression
def test_leak_detector_clears_a_function_that_does_not_leak() -> None:
    """The detector must distinguish, not just flag.

    ``test_detect_memory_leak`` only checks that a leaking function is caught,
    which a detector hard-coded to return True would also pass.
    """
    packets = [_packet() for _ in range(5)]

    def clean(data: Iterable[CSIData]) -> int:
        return sum(1 for _ in data)

    assert not detect_memory_leak(clean, packets, runs=5, threshold=100_000)


@pytest.mark.regression
def test_repeated_processing_does_not_accumulate_memory() -> None:
    """Running the same workload repeatedly must not grow the heap."""
    packets = [_packet() for _ in range(20)]

    def process(data: Iterable[CSIData]) -> float:
        return float(np.mean([packet.amplitude.mean() for packet in data]))

    stats = profile_memory_usage(process, packets, runs=5)
    # Peak should track the mean closely; a steadily growing peak indicates
    # state surviving between runs.
    assert stats["peak_bytes"] < stats["mean_bytes"] * 4, (
        f"peak {stats['peak_bytes']:.0f}B far exceeds mean "
        f"{stats['mean_bytes']:.0f}B across repeats, suggesting accumulation"
    )


# ---------------------------------------------------------------------------
# Measurement invariants: any correct latency report must satisfy these
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_latency_percentiles_are_ordered() -> None:
    """p50 <= p95 <= p99 <= max, and min <= mean <= max."""
    packets = [_packet() for _ in range(10)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    stats = measure_latency(predictor, packets, runs=25)

    assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
    assert stats["p99_ms"] <= stats["max_ms"]
    assert stats["min_ms"] >= 0.0


@pytest.mark.regression
def test_a_slower_predictor_measures_slower() -> None:
    """A relative comparison on one host, immune to absolute machine speed.

    If this fails the harness is not measuring the predictor at all.
    """
    packets = [_packet() for _ in range(5)]

    def fast(packet: CSIData) -> float:
        return float(packet.amplitude[0, 0, 0])

    def slow(packet: CSIData) -> float:
        time.sleep(0.002)
        return float(packet.amplitude[0, 0, 0])

    fast_ms = measure_latency(fast, packets, runs=10, warmup=2)["mean_ms"]
    slow_ms = measure_latency(slow, packets, runs=10, warmup=2)["mean_ms"]

    assert slow_ms > fast_ms, (
        f"a deliberately slowed predictor measured {slow_ms:.3f}ms against "
        f"{fast_ms:.3f}ms for the fast one"
    )


# ---------------------------------------------------------------------------
# Absolute backstops -- generous, and excludable with -m "not slow"
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.regression
def test_trivial_prediction_stays_under_a_generous_ceiling() -> None:
    """Catches order-of-magnitude regressions, not small drifts.

    The ceiling is set roughly 100x above a typical observation so it fails
    only on a genuine collapse, never on runner noise.
    """
    packets = [_packet() for _ in range(5)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    stats = measure_latency(predictor, packets, runs=20, warmup=5)
    assert stats["p95_ms"] < 5.0, (
        f"p95 latency {stats['p95_ms']:.3f}ms for a mean over 30 floats; "
        "expected microseconds"
    )

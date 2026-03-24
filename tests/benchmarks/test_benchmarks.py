"""Tests for benchmark utility functions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn


from benchmarks.accuracy_benchmark import run_accuracy_benchmark  # noqa: E402
from benchmarks.latency_benchmark import measure_latency  # noqa: E402
from benchmarks.memory_benchmark import (  # noqa: E402
    detect_memory_leak,
    measure_memory_usage,
    profile_memory_usage,
)
from benchmarks.performance_report import generate_performance_report  # noqa: E402
from wifi_activity_recognition.hardware.base import CSIData  # noqa: E402


def _make_csi_packet(subcarriers: int = 30) -> CSIData:
    amp = np.random.rand(1, 1, subcarriers).astype(np.float32)
    phase = np.random.rand(1, 1, subcarriers).astype(np.float32)
    return CSIData(
        timestamp=time.time(),
        amplitude=amp,
        phase=phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=subcarriers,
    )


def test_run_accuracy_benchmark_single() -> None:
    """Top-1 and Top-2 accuracy on a single dataset."""

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    model = DummyModel()
    inputs = torch.zeros((10, 4))
    targets = torch.zeros(10, dtype=torch.long)
    dataloader = [(inputs, targets)]

    acc = run_accuracy_benchmark(model, dataloader, topk=(1, 2))
    assert acc[1] == 1.0 and acc[2] == 1.0


def test_run_accuracy_benchmark_multi_device() -> None:
    """Ensure benchmarking works across multiple devices."""

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    model = DummyModel()
    inputs = torch.zeros((10, 4))
    targets = torch.zeros(10, dtype=torch.long)
    loaders = {"set1": [(inputs, targets)]}

    res = run_accuracy_benchmark(model, loaders, devices=["cpu"], aggregate=False)
    assert res["cpu"]["set1"] == 1.0


def test_run_accuracy_benchmark_aggregate() -> None:
    """Overall accuracy computation across datasets."""

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    model = DummyModel()
    inputs = torch.zeros((10, 4))
    targets = torch.zeros(10, dtype=torch.long)
    loaders = {"set1": [(inputs, targets)]}

    res = run_accuracy_benchmark(model, loaders)
    assert res["overall"] == 1.0 and res["datasets"]["set1"] == 1.0


def test_measure_latency() -> None:
    """Latency statistics include percentile values."""
    packets = [_make_csi_packet() for _ in range(5)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    latency = measure_latency(predictor, packets, runs=5)
    assert latency["mean_ms"] >= 0.0 and "p95_ms" in latency


def test_measure_latency_multi_device() -> None:
    """Latency benchmarking across devices returns nested mapping."""
    packets = [_make_csi_packet() for _ in range(3)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    res = measure_latency(predictor, packets, runs=2, devices=["cpu"])
    assert "cpu" in res and "mean_ms" in res["cpu"]


def test_measure_memory_usage() -> None:
    """Peak memory usage is greater than zero."""
    packets = [_make_csi_packet() for _ in range(5)]

    def consumer(data: Iterable[CSIData]) -> List[CSIData]:
        return list(data)

    peak = measure_memory_usage(consumer, packets, optimize=True)
    assert peak > 0


def test_profile_memory_usage() -> None:
    """Profile function returns mean and peak memory usage."""
    packets = [_make_csi_packet() for _ in range(5)]

    def consumer(data: Iterable[CSIData]) -> List[CSIData]:
        return list(data)

    stats = profile_memory_usage(consumer, packets, runs=2, optimize=True)
    assert stats["peak_bytes"] >= stats["mean_bytes"] > 0


def test_detect_memory_leak() -> None:
    """Leak detector flags functions that accumulate data."""
    packets = [_make_csi_packet() for _ in range(5)]
    store: List[CSIData] = []

    def leaky(data: Iterable[CSIData]) -> None:
        store.extend(list(data))

    assert detect_memory_leak(leaky, packets, runs=2, threshold=1)


def test_generate_performance_report(tmp_path: Path) -> None:
    """End-to-end generation of JSON performance report."""

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x)

    model = DummyModel()
    inputs = torch.zeros((10, 4))
    targets = torch.zeros(10, dtype=torch.long)
    loaders = {"set": [(inputs, targets)]}

    packets = [_make_csi_packet() for _ in range(3)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    def consumer(data: Iterable[CSIData]) -> List[CSIData]:
        return list(data)

    path = tmp_path / "report.json"
    report = generate_performance_report(
        model, loaders, predictor, packets, consumer, path
    )
    assert path.exists()
    assert "accuracy" in report and "latency_ms" in report and "memory_bytes" in report
    assert isinstance(report["memory_bytes"], dict)


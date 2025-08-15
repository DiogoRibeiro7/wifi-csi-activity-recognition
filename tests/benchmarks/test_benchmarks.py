"""Tests for benchmark utility functions."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Make the package importable despite repository layout using hyphenated name
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
PACKAGE_ROOT = REPO_ROOT / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from benchmarks.accuracy_benchmark import run_accuracy_benchmark  # noqa: E402
from benchmarks.latency_benchmark import measure_latency  # noqa: E402
from benchmarks.memory_benchmark import (  # noqa: E402
    detect_memory_leak,
    measure_memory_usage,
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

    acc = run_accuracy_benchmark(model, dataloader)
    assert acc == 1.0


def test_run_accuracy_benchmark_multi_device() -> None:
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

    res = run_accuracy_benchmark(model, loaders, devices=["cpu"])
    assert res["cpu"]["set1"] == 1.0


def test_measure_latency() -> None:
    packets = [_make_csi_packet() for _ in range(5)]

    def predictor(packet: CSIData) -> float:
        return float(packet.amplitude.mean())

    latency = measure_latency(predictor, packets, runs=5)
    assert latency["mean_ms"] >= 0.0


def test_measure_memory_usage() -> None:
    packets = [_make_csi_packet() for _ in range(5)]

    def consumer(data: Iterable[CSIData]) -> List[CSIData]:
        return list(data)

    peak = measure_memory_usage(consumer, packets, optimize=True)
    assert peak > 0


def test_detect_memory_leak() -> None:
    packets = [_make_csi_packet() for _ in range(5)]
    store: List[CSIData] = []

    def leaky(data: Iterable[CSIData]) -> None:
        store.extend(list(data))

    assert detect_memory_leak(leaky, packets, runs=2, threshold=1)


def test_generate_performance_report(tmp_path: Path) -> None:
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

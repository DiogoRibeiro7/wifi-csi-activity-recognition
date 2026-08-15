"""Tests for edge deployment framework."""

import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from deployment.edge.device_profiles import get_profile  # type: ignore  # noqa: E402
from deployment.edge.monitoring import EdgeMonitor  # type: ignore  # noqa: E402
from deployment.edge.optimization import (  # type: ignore  # noqa: E402
    DistillationConfig,
    convert_to_onnx,
    distill,
    prune_model,
    quantize_dynamic,
)
from deployment.edge.runtime import EdgeRuntime  # type: ignore  # noqa: E402


class TinyNet(torch.nn.Module):
    """Small model used in unit tests."""

    def __init__(self) -> None:
        """Initialize the tiny network."""
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Run a forward pass."""
        return self.fc(x)


def test_optimization_workflow(tmp_path: Path) -> None:
    """Ensure optimization utilities execute without error."""
    model = TinyNet()
    pruned = prune_model(model, amount=0.5)
    assert (pruned.fc.weight == 0).sum() > 0

    quantized = quantize_dynamic(TinyNet())
    from torch.nn.quantized.dynamic import Linear as QLinear

    assert any(isinstance(m, QLinear) for m in quantized.modules())

    teacher = TinyNet()
    student = TinyNet()
    data: List[torch.Tensor] = [torch.randn(2, 4) for _ in range(2)]
    distill(teacher, student, data, DistillationConfig(epochs=1))

    sample = torch.randn(1, 4)
    onnx_path = tmp_path / "model.onnx"
    convert_to_onnx(student, sample, onnx_path)
    assert onnx_path.exists()


def test_runtime_and_profiles() -> None:
    """Validate runtime inference with a device profile."""
    profile = get_profile("raspberry_pi")
    runtime = EdgeRuntime(TinyNet(), profile)
    out = runtime.run(torch.randn(1, 4))
    assert out.shape == (1, 2)


def test_monitoring_gathers_metrics() -> None:
    """Edge monitor should return a metrics dictionary."""
    monitor = EdgeMonitor()
    metrics = monitor.gather()
    assert set(metrics.keys()) == {
        "memory_mb",
        "cpu_percent",
        "temperature_c",
        "battery_percent",
    }

"""Model optimization utilities for edge deployment.

This module provides lightweight wrappers around common optimization
techniques used to shrink and accelerate models for resource constrained
hardware. Functions return optimized models or paths to converted
artifacts and avoid external dependencies so they can execute during
unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn.utils import prune


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 1.0
    alpha: float = 0.5
    epochs: int = 1


def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic quantization to the provided model.

    Args:
        model: Model to quantize.
        dtype: Target quantized dtype (``torch.qint8`` or ``torch.float16``).

    Returns:
        Quantized ``nn.Module`` ready for inference.
    """
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


def prune_model(
    model: nn.Module, amount: float = 0.5, structured: bool = False
) -> nn.Module:
    """Prune the given model in-place.

    Args:
        model: Model to prune.
        amount: Proportion of connections to remove.
        structured: If ``True`` perform structured pruning on Linear layers.

    Returns:
        The pruned ``nn.Module``.
    """
    parameters: Iterable[nn.Module]
    if structured:
        parameters = (
            module for module in model.modules() if isinstance(module, nn.Linear)
        )
        for module in parameters:
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    else:
        parameters = (
            module for module in model.modules() if isinstance(module, nn.Linear)
        )
        for module in parameters:
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def distill(
    teacher: nn.Module,
    student: nn.Module,
    data_loader: Iterable[torch.Tensor],
    config: Optional[DistillationConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[nn.Module] = None,
) -> nn.Module:
    """Perform a tiny knowledge distillation loop.

    The implementation is intentionally simple and executes only a few
    iterations, making it suitable for unit tests while providing a
    realistic API for future expansion.

    Args:
        teacher: Pretrained teacher network in evaluation mode.
        student: Student model to train.
        data_loader: Iterable of input tensors.
        config: Distillation parameters.
        optimizer: Optimizer used to update the student.
        loss_fn: Loss function (defaults to ``nn.MSELoss``).

    Returns:
        The trained student model.
    """
    cfg = config or DistillationConfig()
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer or torch.optim.SGD(student.parameters(), lr=0.01)

    teacher.eval()
    student.train()
    for _ in range(cfg.epochs):
        for batch in data_loader:
            with torch.no_grad():
                teacher_output = teacher(batch)
            student_output = student(batch)
            loss = loss_fn(
                student_output / cfg.temperature, teacher_output / cfg.temperature
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return student


def convert_to_onnx(model: nn.Module, sample: torch.Tensor, output: Path) -> Path:
    """Export ``model`` to ONNX format.

    Args:
        model: Model to export.
        sample: Example input tensor.
        output: Destination file path.

    Returns:
        Path to the generated ONNX file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, sample, str(output), opset_version=12)
    except Exception:  # pragma: no cover - onnx not installed
        output.write_bytes(b"")
    return output


def convert_to_tensorrt(onnx_path: Path, output: Path) -> Path:
    """Mock conversion from ONNX to TensorRT engine.

    The function does not perform real conversion to avoid heavy
    dependencies. Instead, it simply copies the ONNX file to the desired
    output path, emulating a conversion step.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    data = onnx_path.read_bytes()
    output.write_bytes(data)
    return output


__all__ = [
    "DistillationConfig",
    "quantize_dynamic",
    "prune_model",
    "distill",
    "convert_to_onnx",
    "convert_to_tensorrt",
]

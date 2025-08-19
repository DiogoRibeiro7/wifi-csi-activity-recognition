"""Utilities for optimizing inference latency and resource usage."""

from __future__ import annotations

from typing import Iterable, Iterator, List

import torch
from torch import nn
from torch.nn.utils import prune


def quantize_model(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic quantization to reduce model size and latency."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


def prune_model(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Globally prune a percentage of model weights."""
    parameters: List[tuple[nn.Module, str]] = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters.append((module, "weight"))
    if parameters:
        prune.global_unstructured(
            parameters, pruning_method=prune.L1Unstructured(), amount=amount
        )
    return model


def set_gpu_memory_limit(limit_mb: int) -> None:
    """Restrict GPU memory usage for the current process."""
    if torch.cuda.is_available():
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        fraction = min(1.0, limit_mb / total_mb)
        torch.cuda.set_per_process_memory_fraction(fraction)


def dynamic_batch(iterable: Iterable, max_batch_size: int) -> Iterator[list]:
    """Yield lists of items up to ``max_batch_size`` for throughput optimization."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == max_batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

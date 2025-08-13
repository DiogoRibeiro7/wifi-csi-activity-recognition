"""Utilities for evaluating model accuracy across datasets."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn


def run_accuracy_benchmark(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: Optional[str] = None,
) -> float:
    """Compute classification accuracy of a model over a dataset.

    Args:
        model: PyTorch model returning class logits.
        dataloader: Iterable yielding ``(inputs, targets)`` batches.
        device: Device on which to run the benchmark. If ``None`` the model's
            current device is used.

    Returns:
        Accuracy as a float in the range ``[0.0, 1.0]``.
    """
    model.eval()
    if device is not None:
        model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    if total == 0:
        return 0.0
    return correct / total

"""Utilities for evaluating model accuracy across datasets and devices."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

Loader = Iterable[Tuple[torch.Tensor, torch.Tensor]]


def _accuracy(model: nn.Module, dataloader: Loader, device: Optional[str]) -> float:
    """Internal helper computing accuracy for a single dataloader."""
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


def run_accuracy_benchmark(
    model: nn.Module,
    dataloaders: Union[Loader, Mapping[str, Loader]],
    device: Optional[str] = None,
    devices: Optional[Sequence[str]] = None,
) -> Union[
    float, MutableMapping[str, float], MutableMapping[str, MutableMapping[str, float]]
]:
    """Compute classification accuracy across datasets and devices.

    The function is backward compatible: passing a single dataloader returns a
    float. Supplying a mapping of dataloaders computes per-dataset accuracy. If
    ``devices`` is provided, the benchmark is executed on each device and a
    nested mapping of ``device -> dataset -> accuracy`` is returned.

    Args:
        model: PyTorch model returning class logits.
        dataloaders: Single dataloader or mapping of dataset name to dataloader.
        device: Device to run on when ``devices`` is ``None``.
        devices: Optional sequence of devices for cross-platform evaluation.

    Returns:
        Accuracy value(s) as float or nested mappings depending on input
        parameters.
    """
    if devices is not None:
        results: MutableMapping[str, MutableMapping[str, float]] = {}
        for dev in devices:
            results[dev] = run_accuracy_benchmark(model, dataloaders, device=dev)  # type: ignore[assignment]
        return results

    if isinstance(dataloaders, Mapping):
        return {
            name: _accuracy(model, loader, device)
            for name, loader in dataloaders.items()
        }
    return _accuracy(model, dataloaders, device)

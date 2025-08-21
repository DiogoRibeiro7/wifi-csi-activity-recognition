"""Utilities for evaluating model accuracy across datasets and devices."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

Loader = Iterable[Tuple[torch.Tensor, torch.Tensor]]


def _accuracy(
    model: nn.Module,
    dataloader: Loader,
    device: Optional[str],
    topk: Sequence[int],
) -> Union[float, MutableMapping[int, float]]:
    """Compute top-k accuracy for a single dataloader."""
    model.eval()
    if device is not None:
        model.to(device)

    maxk = max(topk)
    correct_k = {k: 0 for k in topk}
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            outputs = model(inputs)
            _, pred = outputs.topk(maxk, dim=1)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            for k in topk:
                correct_k[k] += correct[:k].reshape(-1).float().sum().item()
            total += targets.numel()
    if total == 0:
        return {k: 0.0 for k in topk} if len(topk) > 1 else 0.0
    if len(topk) == 1:
        return correct_k[topk[0]] / total
    return {k: correct_k[k] / total for k in topk}


def run_accuracy_benchmark(
    model: nn.Module,
    dataloaders: Union[Loader, Mapping[str, Loader]],
    device: Optional[str] = None,
    devices: Optional[Sequence[str]] = None,
    *,
    topk: int | Sequence[int] = 1,
    aggregate: bool = True,
) -> Union[
    float,
    MutableMapping[int, float],
    MutableMapping[str, Union[float, MutableMapping[int, float]]],
    MutableMapping[str, MutableMapping[str, Union[float, MutableMapping[int, float]]]],
]:
    """Compute classification accuracy across datasets and devices.

    The function supports multi-dataset evaluation, top-k accuracy and
    cross-platform testing across several devices. When ``dataloaders`` is a
    mapping, per-dataset results are returned and ``aggregate`` controls whether
    an overall accuracy score is computed across all datasets.

    Args:
        model: PyTorch model returning class logits.
        dataloaders: Single dataloader or mapping of dataset name to dataloader.
        device: Device to run on when ``devices`` is ``None``.
        devices: Optional sequence of devices for cross-platform evaluation.
        topk: Single ``k`` or sequence of ``k`` values for top-k accuracy.
        aggregate: Whether to compute overall accuracy when multiple datasets
            are supplied.

    Returns:
        Accuracy value(s) as float or nested mappings depending on input
        parameters.
    """
    if isinstance(topk, int):
        topk_vals: Sequence[int] = (topk,)
    else:
        topk_vals = tuple(sorted(set(topk)))

    if devices is not None:
        results: MutableMapping[
            str, MutableMapping[str, Union[float, MutableMapping[int, float]]]
        ] = {}
        for dev in devices:
            results[dev] = run_accuracy_benchmark(
                model,
                dataloaders,
                device=dev,
                topk=topk_vals,
                aggregate=aggregate,
            )  # type: ignore[assignment]
        return results

    if isinstance(dataloaders, Mapping):
        dataset_results: MutableMapping[
            str, Union[float, MutableMapping[int, float]]
        ] = {
            name: _accuracy(model, loader, device, topk_vals)
            for name, loader in dataloaders.items()
        }
        if aggregate:
            combined: list[Tuple[torch.Tensor, torch.Tensor]] = []
            for loader in dataloaders.values():
                combined.extend(list(loader))
            overall = _accuracy(model, combined, device, topk_vals)
            return {"overall": overall, "datasets": dataset_results}
        return dataset_results

    return _accuracy(model, dataloaders, device, topk_vals)

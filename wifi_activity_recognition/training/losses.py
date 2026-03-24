"""Custom loss functions for activity recognition."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def cross_entropy_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute standard cross-entropy loss."""
    return nn.functional.cross_entropy(outputs, targets)


def focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
) -> torch.Tensor:
    """Return focal loss to address class imbalance."""
    ce_loss = nn.functional.cross_entropy(outputs, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    if alpha is not None:
        ce_loss = alpha * ce_loss
    return ((1 - pt) ** gamma * ce_loss).mean()


def label_smoothing_loss(
    outputs: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.1
) -> torch.Tensor:
    """Cross entropy with label smoothing."""
    num_classes = outputs.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(outputs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    log_probs = nn.functional.log_softmax(outputs, dim=1)
    return -(true_dist * log_probs).sum(dim=1).mean()


__all__ = ["cross_entropy_loss", "focal_loss", "label_smoothing_loss"]

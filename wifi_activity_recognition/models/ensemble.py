"""Model ensemble utilities for CSI activity recognition.

The ensemble model combines predictions from multiple base models to improve
robustness. It supports any models returning logits for the same set of
classes and aggregates them via simple averaging.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .cnn2d import CNN2DModel
from .cnn3d import CNN3DModel
from .resnet import ResNetSpectrogramModel
from .vision_transformer import VisionTransformerModel


class EnsembleModel(nn.Module):
    """Average ensemble of CNN2D, ResNet, and CNN3D models."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        cnn2d: Optional[nn.Module] = None,
        resnet: Optional[nn.Module] = None,
        cnn3d: Optional[nn.Module] = None,
        vit: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the ensemble model."""
        super().__init__()
        self.cnn2d = cnn2d or CNN2DModel(
            num_classes=num_classes, in_channels=in_channels
        )
        self.resnet = resnet or ResNetSpectrogramModel(
            num_classes=num_classes, in_channels=in_channels, pretrained=False
        )
        self.cnn3d = cnn3d or CNN3DModel(
            num_classes=num_classes, in_channels=in_channels
        )
        self.vit = vit or VisionTransformerModel(
            num_classes=num_classes, in_channels=in_channels
        )

    def forward(self, x2d: torch.Tensor, x3d: torch.Tensor) -> torch.Tensor:
        """Forward pass combining predictions from submodels."""
        logits = [
            self.cnn2d(x2d),
            self.resnet(x2d),
            self.cnn3d(x3d),
            self.vit(x2d),
        ]
        return torch.stack(logits).mean(0)


__all__ = ["EnsembleModel"]

"""ResNet models adapted for CSI spectrogram classification.

This module provides PyTorch and TensorFlow variants of a ResNet architecture
suited for CSI spectrogram inputs. The PyTorch implementation leverages
``torchvision``'s pretrained weights for transfer learning and adapts the first
convolutional layer to accept an arbitrary number of input channels. Both
implementations use global average pooling to accommodate variable input
sizes.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

try:  # Optional TensorFlow import
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover - tensorflow may be unavailable
    tf = None
    keras = None
    layers = None


class ResNetSpectrogramModel(nn.Module):
    """ResNet18 backbone adapted for CSI spectrograms using PyTorch.

    Parameters
    ----------
    num_classes:
        Number of activity classes to predict.
    in_channels:
        Number of input channels (e.g., amplitude, phase).
    pretrained:
        If ``True``, load ImageNet pretrained weights for transfer learning.
    """

    def __init__(
        self, num_classes: int, in_channels: int = 1, pretrained: bool = True
    ) -> None:
        """Initialize the ResNet model."""
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    conv1.weight = nn.Parameter(
                        backbone.conv1.weight.mean(dim=1, keepdim=True).repeat(
                            1, in_channels, 1, 1
                        )
                    )
            backbone.conv1 = conv1
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run a forward pass."""
        return self.model(x)


if keras is not None:

    class ResNetSpectrogramTensorFlowModel(keras.Model):  # type: ignore[misc]
        """TensorFlow variant of :class:`ResNetSpectrogramModel`."""

        def __init__(
            self, num_classes: int, in_channels: int = 1, pretrained: bool = True
        ) -> None:
            """Initialize the TensorFlow ResNet model."""
            super().__init__()
            weights = "imagenet" if pretrained and in_channels == 3 else None
            self.base = tf.keras.applications.ResNet50(
                include_top=False,
                weights=weights,
                input_shape=(None, None, in_channels),
                pooling="avg",
            )
            self.classifier = layers.Dense(num_classes)

        def call(
            self, inputs: tf.Tensor, training: bool = False
        ) -> tf.Tensor:  # type: ignore[override]
            """Run a forward pass."""
            x = self.base(inputs, training=training)
            return self.classifier(x)

else:  # pragma: no cover - TensorFlow is optional

    class ResNetSpectrogramTensorFlowModel:  # type: ignore[too-many-ancestors]
        """Placeholder when TensorFlow is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Initialize placeholder."""
            raise ImportError(
                "TensorFlow is required for ResNetSpectrogramTensorFlowModel"
            )


__all__ = ["ResNetSpectrogramModel", "ResNetSpectrogramTensorFlowModel"]

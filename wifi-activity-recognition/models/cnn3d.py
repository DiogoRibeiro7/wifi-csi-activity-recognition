"""3D CNN models for spatio-temporal CSI sequence classification.

This module implements simple 3D convolutional networks that operate on
sequences of CSI spectrograms. The networks use global average pooling to
handle variable temporal and spatial dimensions.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # Optional TensorFlow import
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover - tensorflow may be unavailable
    tf = None
    keras = None
    layers = None


class CNN3DModel(nn.Module):
    """Simple 3D CNN for CSI sequences using PyTorch."""

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        """Initialize the 3D CNN model."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if keras is not None:

    class CNN3DTensorFlowModel(keras.Model):  # type: ignore[misc]
        """TensorFlow variant of :class:`CNN3DModel`."""

        def __init__(self, num_classes: int, in_channels: int = 1) -> None:
            """Initialize the TensorFlow 3D CNN model."""
            super().__init__()
            self.conv1 = layers.Conv3D(
                32, 3, padding="same", input_shape=(None, None, None, in_channels)
            )
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv3D(64, 3, padding="same")
            self.bn2 = layers.BatchNormalization()
            self.conv3 = layers.Conv3D(128, 3, padding="same")
            self.bn3 = layers.BatchNormalization()
            self.pool = layers.GlobalAveragePooling3D()
            self.classifier = layers.Dense(num_classes)

        def call(
            self, inputs: tf.Tensor, training: bool = False
        ) -> tf.Tensor:  # type: ignore[override]
            """Run a forward pass."""
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = layers.MaxPooling3D(pool_size=2)(x)
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            x = layers.MaxPooling3D(pool_size=2)(x)
            x = self.conv3(x)
            x = self.bn3(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return self.classifier(x)

else:  # pragma: no cover - TensorFlow is optional

    class CNN3DTensorFlowModel:  # type: ignore[too-many-ancestors]
        """Placeholder when TensorFlow is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Initialize placeholder."""
            raise ImportError("TensorFlow is required for CNN3DTensorFlowModel")


__all__ = ["CNN3DModel", "CNN3DTensorFlowModel"]

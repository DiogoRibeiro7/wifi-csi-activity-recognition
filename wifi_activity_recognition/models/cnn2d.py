"""2D CNN models for CSI spectrogram classification.

This module provides both PyTorch and TensorFlow implementations of a simple
convolutional neural network that operates on CSI spectrograms. The models are
designed to handle variable input sizes from different hardware platforms by
using adaptive pooling layers.
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


class CNN2DModel(nn.Module):
    """Simple 2D CNN for CSI spectrograms using PyTorch.

    Parameters
    ----------
    num_classes:
        Number of activity classes to predict.
    in_channels:
        Number of input channels (e.g., amplitude, phase).
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        """Build the convolutional stack and classifier head."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, height, width)``.
        """
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if keras is not None:

    class CNN2DTensorFlowModel(keras.Model):  # type: ignore[misc]
        """TensorFlow variant of :class:`CNN2DModel`.

        Uses similar architecture with adaptive pooling to handle variable input
        shapes.
        """

        def __init__(self, num_classes: int, in_channels: int = 1) -> None:
            """Build the Keras convolutional stack and classifier head."""
            super().__init__()
            self.conv1 = layers.Conv2D(
                32, 3, padding="same", input_shape=(None, None, in_channels)
            )
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv2D(64, 3, padding="same")
            self.bn2 = layers.BatchNormalization()
            self.conv3 = layers.Conv2D(128, 3, padding="same")
            self.bn3 = layers.BatchNormalization()
            self.pool = layers.GlobalAveragePooling2D()
            self.classifier = layers.Dense(num_classes)

        def call(
            self, inputs: tf.Tensor, training: bool = False
        ) -> tf.Tensor:  # type: ignore[override]
            """Run a TensorFlow forward pass for CSI spectrogram inputs.

            Args:
                inputs: Tensor of shape ``(batch, height, width, channels)``.
                training: Whether batch normalization layers should update stats.

            Returns:
                Logit tensor with shape ``(batch, num_classes)``.
            """
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
            x = self.conv3(x)
            x = self.bn3(x, training=training)
            x = tf.nn.relu(x)
            x = self.pool(x)
            return self.classifier(x)

else:  # pragma: no cover - TensorFlow is optional

    class CNN2DTensorFlowModel:  # type: ignore[too-many-ancestors]
        """Placeholder when TensorFlow is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            """Refuse construction: the TensorFlow backend is unavailable."""
            raise ImportError("TensorFlow is required for CNN2DTensorFlowModel")


__all__ = ["CNN2DModel", "CNN2DTensorFlowModel"]

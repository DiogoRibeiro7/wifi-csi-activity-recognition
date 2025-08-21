"""Attention-augmented 3D CNN for spatio-temporal CSI analysis.

This module implements an advanced 3D CNN that combines convolutional
feature extractors with lightweight spatial and temporal attention
mechanisms. It exposes both PyTorch and TensorFlow implementations and
uses adaptive pooling to handle variable input sizes produced by
heterogeneous WiFi hardware.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

try:  # Optional TensorFlow import
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:  # pragma: no cover - TensorFlow may be unavailable
    tf = None
    keras = None
    layers = None


class _SpatialAttention3D(nn.Module):
    """Compute spatial attention across antenna/subcarrier dimensions."""

    def __init__(self, kernel_size: Tuple[int, int] = (7, 7)) -> None:
        super().__init__()
        padding = (0, kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv3d(
            2, 1, kernel_size=(1, *kernel_size), padding=padding, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, mx], dim=1)
        attn = self.conv(attn)
        return x * self.sigmoid(attn)


class _TemporalAttention3D(nn.Module):
    """Compute temporal attention across sequence dimension."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, H, W)
        n, c, t, h, w = x.shape
        x_avg = x.mean(dim=[3, 4])  # (N, C, T)
        x_avg = x_avg.transpose(1, 2)  # (N, T, C)
        attn = self.fc2(torch.relu(self.fc1(x_avg)))  # (N, T, C)
        attn = attn.mean(dim=2, keepdim=True)  # (N, T, 1)
        weights = self.softmax(attn).transpose(1, 2)  # (N, 1, T)
        weights = weights.view(n, 1, t, 1, 1)
        return x * weights


class AttentionCNN3DModel(nn.Module):
    """3D CNN with spatial and temporal attention (PyTorch)."""

    def __init__(
        self, num_classes: int, in_channels: int = 1, dropout: float = 0.3
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.MaxPool3d((1, 1, 2)),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.spatial_attn = _SpatialAttention3D()
        self.temporal_attn = _TemporalAttention3D(128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        x = self.features(x)
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        x = self.pool(x)
        return self.classifier(x)


if keras is not None:

    class _KerasSpatialAttention(layers.Layer):  # type: ignore[misc]
        """Keras implementation of spatial attention."""

        def __init__(self, kernel_size: Tuple[int, int] = (7, 7)) -> None:
            super().__init__()
            self.avg_pool = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=4, keepdims=True)
            )
            self.max_pool = layers.Lambda(
                lambda x: tf.reduce_max(x, axis=4, keepdims=True)
            )
            self.conv = layers.Conv3D(
                1, (1, *kernel_size), padding="same", use_bias=False
            )

        def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
            avg = self.avg_pool(inputs)
            mx = self.max_pool(inputs)
            x = tf.concat([avg, mx], axis=4)
            return inputs * tf.nn.sigmoid(self.conv(x))

    class _KerasTemporalAttention(layers.Layer):  # type: ignore[misc]
        """Keras implementation of temporal attention."""

        def __init__(self, channels: int, reduction: int = 16) -> None:
            super().__init__()
            reduced = max(1, channels // reduction)
            self.fc1 = layers.Dense(reduced, activation="relu")
            self.fc2 = layers.Dense(1)

        def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
            # inputs: (N, T, H, W, C)
            x = tf.reduce_mean(inputs, axis=[2, 3])  # (N, T, C)
            attn = self.fc2(self.fc1(x))  # (N, T, 1)
            weights = tf.nn.softmax(attn, axis=1)
            weights = tf.reshape(weights, (-1, inputs.shape[1], 1, 1, 1))
            return inputs * weights

    class AttentionCNN3DTensorFlowModel(keras.Model):  # type: ignore[misc]
        """TensorFlow variant of :class:`AttentionCNN3DModel`."""

        def __init__(
            self, num_classes: int, in_channels: int = 1, dropout: float = 0.3
        ) -> None:
            """Initialize the TensorFlow attention CNN3D model."""
            super().__init__()
            self.conv1 = layers.Conv3D(
                32, 3, padding="same", input_shape=(None, None, None, in_channels)
            )
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv3D(64, 3, padding="same")
            self.bn2 = layers.BatchNormalization()
            self.drop = layers.SpatialDropout3D(dropout)
            self.conv3 = layers.Conv3D(128, 3, padding="same")
            self.bn3 = layers.BatchNormalization()
            self.s_attn = _KerasSpatialAttention()
            self.t_attn = _KerasTemporalAttention(128)
            self.pool = layers.GlobalAveragePooling3D()
            self.classifier = layers.Dense(num_classes)

        def call(
            self, inputs: tf.Tensor, training: bool = False
        ) -> tf.Tensor:  # type: ignore[override]
            """Run a forward pass."""
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            x = tf.nn.relu(x)
            x = self.drop(x, training=training)
            x = layers.MaxPooling3D(pool_size=(1, 1, 2))(x)
            x = self.conv3(x)
            x = self.bn3(x, training=training)
            x = tf.nn.relu(x)
            x = self.s_attn(x)
            x = self.t_attn(x)
            x = self.pool(x)
            return self.classifier(x)

else:  # pragma: no cover - TensorFlow is optional

    class AttentionCNN3DTensorFlowModel:  # type: ignore[too-many-ancestors]
        """Placeholder when TensorFlow is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            """Initialize placeholder."""
            raise ImportError(
                "TensorFlow is required for AttentionCNN3DTensorFlowModel"
            )


__all__ = ["AttentionCNN3DModel", "AttentionCNN3DTensorFlowModel"]

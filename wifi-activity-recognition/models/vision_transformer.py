"""Vision Transformer model tailored for CSI spectrograms."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
    from tensorflow import keras
except Exception:  # pragma: no cover - handled in tests
    tf = None  # type: ignore
    keras = None  # type: ignore


class TransformerEncoderBlock(nn.Module):
    """Single encoder block with MSA and MLP."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply attention and feed-forward network."""
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + residual


def _get_2d_sincos_pos_embed(
    embed_dim: int, grid_h: int, grid_w: int, cls_token: bool = True
) -> np.ndarray:
    """Create 2D sine-cosine positional embeddings."""

    def _get_1d(pos: np.ndarray, dim: int) -> np.ndarray:
        assert dim % 2 == 0
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / (10000**omega)
        out = np.einsum("n,d->nd", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_w_vec = np.arange(grid_w, dtype=np.float32)
    grid_h_vec = np.arange(grid_h, dtype=np.float32)
    grid = np.meshgrid(grid_w_vec, grid_h_vec)  # (2, H, W)
    grid = np.stack(grid, axis=-1).reshape(-1, 2)
    pos_embed = np.concatenate(
        [
            _get_1d(grid[:, 0], embed_dim // 2),
            _get_1d(grid[:, 1], embed_dim // 2),
        ],
        axis=1,
    )
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


class VisionTransformerModel(nn.Module):
    """Vision Transformer for CSI spectrograms.

    This PyTorch implementation accepts spectrograms of arbitrary size by
    padding them to match the patch size. Two-dimensional sine-cos positional
    embeddings encode frequency and time axes. The model can operate in
    classification mode or return per-patch predictions for sequence-to-sequence
    tasks.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        patch_size: int = 5,
        dim: int = 128,
        depth: int = 4,
        heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        seq_to_seq: bool = False,
        pretrained_state_dict: Optional[dict] = None,
    ) -> None:
        """Initialize the Vision Transformer."""
        super().__init__()
        self.seq_to_seq = seq_to_seq
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(dim, heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(dim, num_classes)
        if pretrained_state_dict is not None:
            self.load_state_dict(pretrained_state_dict, strict=False)

    def _pos_embed(self, h: int, w: int) -> torch.Tensor:
        pe = _get_2d_sincos_pos_embed(self.dim, h, w, cls_token=True)
        return torch.from_numpy(pe).float().unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute logits or sequence outputs.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, channels, freq, time)``.
        """
        b = x.shape[0]
        pad_h = (-x.shape[2]) % self.patch_size
        pad_w = (-x.shape[3]) % self.patch_size
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        pos_embed = self._pos_embed(h, w).to(x.device)
        x = x + pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        if self.seq_to_seq:
            return self.head(x[:, 1:])
        return self.head(x[:, 0])


if tf is not None:

    class TransformerEncoderBlockTF(keras.layers.Layer):  # pragma: no cover - TF path
        """TensorFlow encoder block with attention and MLP."""

        def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.attn = keras.layers.MultiHeadAttention(
                num_heads=heads, key_dim=dim // heads, dropout=dropout
            )
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.mlp = keras.Sequential(
                [
                    keras.layers.Dense(mlp_dim, activation=keras.activations.gelu),
                    keras.layers.Dropout(dropout),
                    keras.layers.Dense(dim),
                    keras.layers.Dropout(dropout),
                ]
            )

        def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
            residual = x
            x = self.norm1(x)
            x = self.attn(x, x, training=training)
            x = residual + x
            residual = x
            x = self.norm2(x)
            x = self.mlp(x, training=training)
            return x + residual

    class VisionTransformerTensorFlowModel(keras.Model):  # pragma: no cover - TF path
        """TensorFlow Vision Transformer for CSI spectrograms."""

        def __init__(
            self,
            num_classes: int,
            in_channels: int = 1,
            patch_size: int = 5,
            dim: int = 128,
            depth: int = 4,
            heads: int = 8,
            mlp_dim: int = 256,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            seq_to_seq: bool = False,
        ) -> None:
            """Initialize the TensorFlow Vision Transformer."""
            super().__init__()
            self.seq_to_seq = seq_to_seq
            self.patch_size = patch_size
            self.dim = dim
            self.patch_embed = keras.layers.Conv2D(
                dim, patch_size, strides=patch_size, padding="valid"
            )
            self.cls_token = self.add_weight(
                "cls_token", shape=(1, 1, dim), initializer="zeros"
            )
            self.dropout = keras.layers.Dropout(emb_dropout)
            self.blocks = [
                TransformerEncoderBlockTF(dim, heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
            self.head = keras.layers.Dense(num_classes)

        def _pos_embed(self, h: int, w: int) -> tf.Tensor:
            """Return positional embeddings for a ``h``x``w`` grid."""
            pe = _get_2d_sincos_pos_embed(self.dim, h, w, cls_token=True)
            return tf.convert_to_tensor(pe, dtype=tf.float32)[tf.newaxis, ...]

        def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
            """Run the forward pass.

            Parameters
            ----------
            x:
                Input tensor of shape ``(batch, freq, time, channels)``.
            training:
                Whether to enable dropout layers.
            """
            b = tf.shape(x)[0]
            pad_h = (-tf.shape(x)[1]) % self.patch_size
            pad_w = (-tf.shape(x)[2]) % self.patch_size
            if pad_h or pad_w:
                x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            x = self.patch_embed(x)
            h = tf.shape(x)[1]
            w = tf.shape(x)[2]
            x = tf.reshape(x, [b, -1, self.dim])
            cls = tf.broadcast_to(self.cls_token, [b, 1, self.dim])
            x = tf.concat([cls, x], axis=1)
            pos = self._pos_embed(h, w)
            x = x + pos
            x = self.dropout(x, training=training)
            for blk in self.blocks:
                x = blk(x, training=training)
            if self.seq_to_seq:
                return self.head(x[:, 1:], training=training)
            return self.head(x[:, 0], training=training)

else:  # pragma: no cover - TF not available

    class VisionTransformerTensorFlowModel:  # type: ignore
        """Stub TensorFlow model when TensorFlow is not installed."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D107
            raise ImportError(
                "TensorFlow is required for VisionTransformerTensorFlowModel"
            )


__all__ = ["VisionTransformerModel", "VisionTransformerTensorFlowModel"]

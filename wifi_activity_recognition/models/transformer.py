"""Transformer-based classifier for CSI sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Self-attention is permutation-invariant, so without this a Transformer
    cannot tell ``(x_1, ..., x_T)`` from ``(x_T, ..., x_1)``. For activity
    recognition that distinction is the signal itself -- sitting down and
    standing up are largely the same frames in the opposite order.

    The encoding is deterministic and recomputed at construction, so the buffer
    is registered non-persistently: it stays out of ``state_dict`` and
    checkpoints written before this module existed still load.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        """Build the encoding table for sequences up to ``max_len``."""
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        # For odd d_model the cosine half is one column shorter than div_term.
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Add positional information to ``(batch, seq_len, d_model)`` input."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_len {self.max_len}; "
                "construct the model with a larger max_len"
            )
        return self.dropout(x + self.pe[:, :seq_len])


class TransformerModel(nn.Module):
    """Transformer encoder classifier for CSI sequences."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        """Initialize the Transformer classifier."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute logits for a batch of sequences.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, seq_len, input_dim)``.
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


__all__ = ["PositionalEncoding", "TransformerModel"]

"""Transformer-based classifier for CSI sequences."""

from __future__ import annotations

import torch
from torch import nn


class TransformerModel(nn.Module):
    """Simple Transformer encoder classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Transformer classifier."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
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
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


__all__ = ["TransformerModel"]

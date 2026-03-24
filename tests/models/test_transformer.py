"""Tests for transformer-based model."""

from pathlib import Path

import torch


from wifi_activity_recognition.models.transformer import (  # type: ignore  # noqa: E402
    TransformerModel,
)


def test_forward_shape() -> None:
    """Model produces logits with correct shape."""
    model = TransformerModel(input_dim=32, num_classes=4)
    x = torch.randn(2, 15, 32)
    out = model(x)
    assert out.shape == (2, 4)


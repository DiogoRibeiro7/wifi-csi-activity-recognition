"""Tests for transformer-based model."""

import sys
import types
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.models.transformer import (  # type: ignore  # noqa: E402
    TransformerModel,
)


def test_forward_shape() -> None:
    """Model produces logits with correct shape."""
    model = TransformerModel(input_dim=32, num_classes=4)
    x = torch.randn(2, 15, 32)
    out = model(x)
    assert out.shape == (2, 4)


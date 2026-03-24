"""Tests for model factory."""

import sys
import types
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.models import (  # type: ignore  # noqa: E402
    AttentionCNN3DModel,
    CNN2DModel,
    CNN3DModel,
    EnsembleModel,
    ResNetSpectrogramModel,
    TransformerModel,
    VisionTransformerModel,
    create_model,
)


def test_factory_creates_models() -> None:
    """Factory constructs registered models and runs forward passes."""
    assert isinstance(create_model("cnn2d", num_classes=2), CNN2DModel)
    assert isinstance(
        create_model("resnet", num_classes=2, pretrained=False),
        ResNetSpectrogramModel,
    )
    assert isinstance(create_model("cnn3d", num_classes=2), CNN3DModel)
    assert isinstance(
        create_model("attention_cnn3d", num_classes=2), AttentionCNN3DModel
    )
    model = create_model("ensemble", num_classes=2)
    assert isinstance(model, EnsembleModel)
    x2d = torch.randn(1, 1, 30, 50)
    x3d = torch.randn(1, 1, 8, 30, 50)
    out = model(x2d, x3d)
    assert out.shape == (1, 2)
    transformer = create_model("transformer", input_dim=64, num_classes=2)
    assert isinstance(transformer, TransformerModel)
    x = torch.randn(1, 10, 64)
    out_t = transformer(x)
    assert out_t.shape == (1, 2)
    vit = create_model("vit", num_classes=2)
    assert isinstance(vit, VisionTransformerModel)
    x_vit = torch.randn(1, 1, 30, 50)
    out_vit = vit(x_vit)
    assert out_vit.shape == (1, 2)


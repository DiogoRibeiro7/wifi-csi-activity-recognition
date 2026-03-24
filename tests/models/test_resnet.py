"""Tests for ResNet model."""

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.models.resnet import (  # type: ignore  # noqa: E402
    ResNetSpectrogramModel,
)


@pytest.mark.parametrize("subcarriers", [30, 64, 128])
def test_resnet_forward_shapes(subcarriers: int) -> None:
    """Test that ResNet produces expected logits shape."""
    model = ResNetSpectrogramModel(num_classes=5, pretrained=False)
    x = torch.randn(2, 1, subcarriers, 200)
    out = model(x)
    assert out.shape == (2, 5)


def test_resnet_training_step_decreases_loss() -> None:
    """Test that training step reduces loss for ResNet."""
    torch.manual_seed(0)
    model = ResNetSpectrogramModel(num_classes=3, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(10, 1, 30, 50)
    y = torch.randint(0, 3, (10,))
    loss1 = criterion(model(x), y)
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    loss2 = criterion(model(x), y)
    assert loss2.item() < loss1.item()


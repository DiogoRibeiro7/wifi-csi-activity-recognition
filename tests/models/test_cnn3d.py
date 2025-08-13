"""Tests for 3D CNN model."""

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.models.cnn3d import (  # type: ignore  # noqa: E402
    CNN3DModel,
)


@pytest.mark.parametrize("depth", [8, 16])
def test_cnn3d_forward_shapes(depth: int) -> None:
    """Test that CNN3D outputs correct shape."""
    model = CNN3DModel(num_classes=4)
    x = torch.randn(2, 1, depth, 30, 50)
    out = model(x)
    assert out.shape == (2, 4)


def test_cnn3d_training_step_decreases_loss() -> None:
    """Training step should reduce loss for CNN3D."""
    model = CNN3DModel(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(6, 1, 8, 30, 50)
    y = torch.randint(0, 3, (6,))
    loss1 = criterion(model(x), y)
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    loss2 = criterion(model(x), y)
    assert loss2.item() < loss1.item()

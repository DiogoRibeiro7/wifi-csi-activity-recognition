"""Tests for attention-based 3D CNN model."""
import resource
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

from wifi_activity_recognition.models import (  # type: ignore  # noqa: E402
    advanced_cnn3d as adv_cnn3d,
)
from wifi_activity_recognition.models import (  # type: ignore  # noqa: E402
    cnn3d as base_cnn3d,
)

AttentionCNN3DModel = adv_cnn3d.AttentionCNN3DModel
CNN3DModel = base_cnn3d.CNN3DModel


@pytest.mark.parametrize("shape", [(2, 1, 8, 3, 30), (2, 1, 10, 2, 64)])
def test_attention_cnn3d_forward_shapes(shape: tuple[int, int, int, int, int]) -> None:
    """Model should handle variable input sizes."""
    model = AttentionCNN3DModel(num_classes=5)
    x = torch.randn(shape)
    out = model(x)
    assert out.shape == (shape[0], 5)


def test_attention_cnn3d_gradients() -> None:
    """Gradients should flow through the network."""
    model = AttentionCNN3DModel(num_classes=3)
    x = torch.randn(4, 1, 8, 3, 30)
    y = torch.randint(0, 3, (4,))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(x), y)
    loss.backward()
    grad_exists = any(p.grad is not None for p in model.parameters())
    assert grad_exists


def test_attention_cnn3d_memory_usage() -> None:
    """Forward/backward pass should not excessively increase memory usage."""
    model = AttentionCNN3DModel(num_classes=3)
    x = torch.randn(2, 1, 8, 3, 30)
    y = torch.randint(0, 3, (2,))
    criterion = nn.CrossEntropyLoss()
    start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    loss = criterion(model(x), y)
    loss.backward()
    end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert (end - start) < 200_000  # roughly <200MB increase


def test_attention_cnn3d_matches_baseline_shape() -> None:
    """Output shape should match baseline CNN3D model on typical input."""
    baseline = CNN3DModel(num_classes=4)
    advanced = AttentionCNN3DModel(num_classes=4)
    x = torch.randn(2, 1, 8, 30, 30)
    out_base = baseline(x)
    out_adv = advanced(x)
    assert out_base.shape == out_adv.shape

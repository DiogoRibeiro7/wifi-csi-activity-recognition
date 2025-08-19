"""Tests for loss and metric utilities."""

import sys
import types
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.training import (  # type: ignore  # noqa: E402
    classification_metrics,
    cross_entropy_loss,
    focal_loss,
    label_smoothing_loss,
)


def test_loss_functions():
    """Check that custom loss functions return valid values."""
    outputs = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1])
    ce = cross_entropy_loss(outputs, targets)
    assert torch.allclose(ce, torch.nn.functional.cross_entropy(outputs, targets))
    fl = focal_loss(outputs, targets)
    ls = label_smoothing_loss(outputs, targets)
    assert fl.item() > 0
    assert ls.item() > 0


def test_classification_metrics():
    """Verify computation of classification metrics."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 1, 1]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    for key in ["precision", "recall", "f1"]:
        assert 0.0 <= metrics[key] <= 1.0

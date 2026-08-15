"""Tests for ensemble model."""

import torch

from wifi_activity_recognition.models.ensemble import (  # type: ignore  # noqa: E402
    EnsembleModel,
)


def test_ensemble_forward() -> None:
    """Ensemble averages predictions from submodels."""
    model = EnsembleModel(num_classes=3)
    x2d = torch.randn(4, 1, 30, 50)
    x3d = torch.randn(4, 1, 8, 30, 50)
    out = model(x2d, x3d)
    assert out.shape == (4, 3)

"""Tests for multi-modal fusion strategies."""

import numpy as np

from wifi_activity_recognition.multimodal.fusion_strategies import (
    early_fusion,
    hybrid_attention_fusion,
    late_fusion,
    uncertainty_aware_fusion,
)


def test_early_fusion_concatenates() -> None:
    """Early fusion should concatenate feature arrays."""
    a = np.ones(3)
    b = np.zeros(2)
    fused = early_fusion([a, b])
    assert fused.shape == (5,)
    assert np.allclose(fused[:3], 1) and np.allclose(fused[3:], 0)


def test_late_fusion_weighted_average() -> None:
    """Late fusion should compute a weighted average of predictions."""
    preds = [np.array([0.6, 0.4]), np.array([0.3, 0.7])]
    fused = late_fusion(preds, weights=[0.75, 0.25])
    expected = 0.75 * preds[0] + 0.25 * preds[1]
    assert np.allclose(fused, expected)


def test_hybrid_attention_prefers_stronger_feature() -> None:
    """Hybrid attention should favor features with larger norms."""
    feats = [np.ones(3), np.zeros(3)]
    preds = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    fused = hybrid_attention_fusion(feats, preds)
    assert fused[0] > fused[1]


def test_uncertainty_aware_fusion() -> None:
    """Uncertainty-aware fusion should weight inverse to variance."""
    preds = [np.array([0.5, 0.5]), np.array([0.2, 0.8])]
    fused = uncertainty_aware_fusion(preds, uncertainties=[0.1, 0.5])
    expected = ((1 / 0.01) * preds[0] + (1 / 0.25) * preds[1]) / (1 / 0.01 + 1 / 0.25)
    assert np.allclose(fused, expected)

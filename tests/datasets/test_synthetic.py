"""Tests for synthetic CSI data generation."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    generate_synthetic_csi,
)


def test_generate_synthetic_csi_shapes():
    """Validate shapes and label range of synthetic generator."""
    data, labels = generate_synthetic_csi(
        num_samples=8,
        num_subcarriers=4,
        num_antennas=2,
        num_classes=3,
        random_state=0,
    )
    assert data.shape == (8, 2, 4)
    assert labels.shape == (8,)
    assert set(labels) <= {0, 1, 2}


def test_generate_synthetic_csi_dtype():
    """Generator respects requested output dtype."""
    data, _ = generate_synthetic_csi(4, 4, dtype=np.float64)
    assert data.dtype == np.float64


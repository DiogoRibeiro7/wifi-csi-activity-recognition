"""Tests for camera and CSI fusion."""

import numpy as np

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.multimodal.camera_integration import CameraFusion


def _csi() -> CSIData:
    """Create a minimal CSI sample for testing."""
    amp = np.ones((1, 1, 4))
    phase = np.zeros((1, 1, 4))
    return CSIData(0.0, amp, phase, 5.0, 20.0, 1, 1, 4)


def test_camera_fusion_basic() -> None:
    """Fusing aligned camera frames returns concatenated features."""
    csi = _csi()
    frame = np.ones((64, 64))
    fusion = CameraFusion()
    fused = fusion.fuse(csi, frame, frame_ts=0.0)
    assert fused.shape[0] == csi.amplitude.size + 64 * 64


def test_camera_fusion_missing_frame() -> None:
    """Missing frame yields zero visual features."""
    csi = _csi()
    fusion = CameraFusion()
    fused = fusion.fuse(csi, None)
    assert np.allclose(fused[-64 * 64 :], 0)

"""Tests for IMU and CSI fusion."""

import numpy as np

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.multimodal.imu_integration import IMUFusion


def _csi() -> CSIData:
    """Create a minimal CSI sample for testing."""
    amp = np.ones((1, 1, 4))
    phase = np.zeros((1, 1, 4))
    return CSIData(0.0, amp, phase, 5.0, 20.0, 1, 1, 4)


def test_imu_fusion_basic() -> None:
    """Fusing aligned IMU data returns concatenated features."""
    csi = _csi()
    imu = np.ones((10, 3))
    fusion = IMUFusion()
    fused = fusion.fuse(csi, imu, imu_ts=0.0)
    assert fused.shape[0] == csi.amplitude.size + 6


def test_imu_fusion_missing_data() -> None:
    """Missing IMU data yields zero features."""
    csi = _csi()
    fusion = IMUFusion()
    fused = fusion.fuse(csi, None)
    assert np.allclose(fused[-6:], 0)

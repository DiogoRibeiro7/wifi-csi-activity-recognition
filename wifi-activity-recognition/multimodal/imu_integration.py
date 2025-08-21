"""Utilities for integrating IMU data with WiFi CSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hardware.base import CSIData
from .fusion_strategies import early_fusion


@dataclass
class IMUFusion:
    """Fuse CSI packets with inertial measurement unit (IMU) readings."""

    tolerance: float = 0.05

    def _extract_features(self, imu: np.ndarray) -> np.ndarray:
        """Compute mean and standard deviation for each axis."""
        mean = imu.mean(axis=0)
        std = imu.std(axis=0)
        return np.concatenate([mean, std])

    def fuse(
        self, csi: CSIData, imu: Optional[np.ndarray], imu_ts: Optional[float] = None
    ) -> np.ndarray:
        """Fuse CSI and IMU features with timestamp alignment."""
        if (
            imu is None
            or imu_ts is None
            or abs(csi.timestamp - imu_ts) > self.tolerance
        ):
            imu_feat = np.zeros(6)
        else:
            imu_feat = self._extract_features(imu)
        csi_feat = csi.amplitude.flatten()
        return early_fusion([csi_feat, imu_feat])

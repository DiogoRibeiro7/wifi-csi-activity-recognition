"""Utilities for integrating camera data with WiFi CSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..hardware.base import CSIData
from .fusion_strategies import early_fusion


@dataclass
class CameraFusion:
    """Fuse CSI packets with privacy-preserving visual features.

    Parameters:
        frame_size: Spatial resolution used for feature extraction. Incoming
            frames are spatially averaged to this size which both reduces
            dimensionality and removes personally identifiable details.
        tolerance: Maximum allowed timestamp difference between CSI and camera
            data before the frame is considered outdated.
    """

    frame_size: Tuple[int, int] = (64, 64)
    tolerance: float = 0.05

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Downsample a frame via mean pooling to preserve privacy."""
        if frame.ndim == 3:  # convert RGB to grayscale
            frame = frame.mean(axis=2)
        h, w = self.frame_size
        sh = max(frame.shape[0] // h, 1)
        sw = max(frame.shape[1] // w, 1)
        small = frame[: h * sh, : w * sw].reshape(h, sh, w, sw).mean(axis=(1, 3))
        return small.flatten()

    def fuse(
        self,
        csi: CSIData,
        frame: Optional[np.ndarray],
        frame_ts: Optional[float] = None,
    ) -> np.ndarray:
        """Fuse CSI and camera features with basic temporal alignment.

        If ``frame`` is ``None`` or the timestamp difference exceeds
        ``tolerance`` seconds, a zero vector is used for the visual features.
        """
        if (
            frame is None
            or frame_ts is None
            or abs(csi.timestamp - frame_ts) > self.tolerance
        ):
            visual_feat = np.zeros(self.frame_size[0] * self.frame_size[1])
        else:
            visual_feat = self._extract_features(frame)
        csi_feat = csi.amplitude.flatten()
        return early_fusion([csi_feat, visual_feat])

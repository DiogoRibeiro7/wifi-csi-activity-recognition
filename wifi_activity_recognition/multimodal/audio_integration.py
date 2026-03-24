"""Utilities for integrating audio data with WiFi CSI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hardware.base import CSIData
from .fusion_strategies import early_fusion


@dataclass
class AudioFusion:
    """Fuse CSI packets with audio waveform features."""

    tolerance: float = 0.05

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-energy and zero-crossing rate of an audio segment."""
        energy = np.log(np.sum(audio**2) + 1e-8)
        zero_cross = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
        return np.array([energy, zero_cross])

    def fuse(
        self,
        csi: CSIData,
        audio: Optional[np.ndarray],
        audio_ts: Optional[float] = None,
    ) -> np.ndarray:
        """Fuse CSI and audio features with timestamp alignment."""
        if (
            audio is None
            or audio_ts is None
            or abs(csi.timestamp - audio_ts) > self.tolerance
        ):
            audio_feat = np.zeros(2)
        else:
            audio_feat = self._extract_features(audio)
        csi_feat = csi.amplitude.flatten()
        return early_fusion([csi_feat, audio_feat])

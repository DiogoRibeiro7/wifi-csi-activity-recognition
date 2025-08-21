"""Tests for audio and CSI fusion."""

import numpy as np

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.multimodal.audio_integration import AudioFusion


def _csi() -> CSIData:
    """Create a minimal CSI sample for testing."""
    amp = np.ones((1, 1, 4))
    phase = np.zeros((1, 1, 4))
    return CSIData(0.0, amp, phase, 5.0, 20.0, 1, 1, 4)


def test_audio_fusion_basic() -> None:
    """Fusing aligned audio frames returns concatenated features."""
    csi = _csi()
    audio = np.sin(np.linspace(0, np.pi * 2, 160))
    fusion = AudioFusion()
    fused = fusion.fuse(csi, audio, audio_ts=0.0)
    assert fused.shape[0] == csi.amplitude.size + 2


def test_audio_fusion_missing_data() -> None:
    """Missing audio data yields zero features."""
    csi = _csi()
    fusion = AudioFusion()
    fused = fusion.fuse(csi, None)
    assert np.allclose(fused[-2:], 0)

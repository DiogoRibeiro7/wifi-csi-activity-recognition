"""Multi-modal sensor fusion utilities."""

from .audio_integration import AudioFusion
from .camera_integration import CameraFusion
from .fusion_strategies import (
    early_fusion,
    hybrid_attention_fusion,
    late_fusion,
    uncertainty_aware_fusion,
)
from .imu_integration import IMUFusion

__all__ = [
    "early_fusion",
    "late_fusion",
    "hybrid_attention_fusion",
    "uncertainty_aware_fusion",
    "CameraFusion",
    "IMUFusion",
    "AudioFusion",
]

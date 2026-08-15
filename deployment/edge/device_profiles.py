"""Device specific optimization profiles for edge deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class DeviceProfile:
    """Configuration describing an edge device."""

    name: str
    backend: str
    quantization: str
    pruning: float
    extra: Dict[str, float]


DEVICE_PROFILES: Dict[str, DeviceProfile] = {
    "raspberry_pi": DeviceProfile(
        name="Raspberry Pi",
        backend="onnx",
        quantization="int8",
        pruning=0.3,
        extra={"memory_limit_mb": 256},
    ),
    "nvidia_jetson": DeviceProfile(
        name="NVIDIA Jetson",
        backend="tensorrt",
        quantization="fp16",
        pruning=0.2,
        extra={"memory_limit_mb": 2048},
    ),
    "android": DeviceProfile(
        name="Android Device",
        backend="onnx",
        quantization="int8",
        pruning=0.4,
        extra={"memory_limit_mb": 512},
    ),
    "ios": DeviceProfile(
        name="iOS Device",
        backend="onnx",
        quantization="fp16",
        pruning=0.4,
        extra={"memory_limit_mb": 512},
    ),
    "custom": DeviceProfile(
        name="Custom Embedded",
        backend="onnx",
        quantization="int8",
        pruning=0.5,
        extra={"memory_limit_mb": 128},
    ),
}


def get_profile(name: str) -> DeviceProfile:
    """Return the device profile for ``name``.

    Raises ``KeyError`` if the profile is not defined.
    """
    return DEVICE_PROFILES[name]


__all__ = ["DeviceProfile", "DEVICE_PROFILES", "get_profile"]

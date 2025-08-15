"""
Hardware abstraction layer for WiFi CSI devices.

This module provides a unified interface for working with different WiFi hardware
platforms that support Channel State Information (CSI) extraction.

Supported Hardware:
- Intel 5300 NIC (research standard)
- ESP32 with CSI capability
- Atheros AR9300 (legacy research)
- Qualcomm platforms (commercial devices)
- Broadcom chipsets (routers)
- MediaTek platforms (emerging)

Basic Usage:
    >>> from wifi_activity_recognition.hardware import CSIReader
    >>>
    >>> # Create reader for specific hardware
    >>> reader = CSIReader('esp32', config={
    ...     'sampling_rate': 100,
    ...     'channel': 6
    ... })
    >>>
    >>> # Connect and stream data
    >>> with reader:
    ...     for csi_data in reader.stream():
    ...         print(f"CSI shape: {csi_data.shape}")
"""

import sys as _sys

from .base import (
    CSIData,
    CSIReaderBase,
    HardwareConfig,
    HardwareFactory,
    normalize_csi_amplitude,
    validate_csi_data,
)

_parent = _sys.modules.get("wifi_activity_recognition")
if _parent is not None:
    setattr(_parent, "hardware", _sys.modules[__name__])

# Import available hardware drivers
try:
    from .intel5300 import Intel5300Reader

    HardwareFactory.register_driver("intel_5300", Intel5300Reader)
    HardwareFactory.register_driver("intel5300", Intel5300Reader)  # Alternative name
except ImportError:
    pass

try:
    from .esp32 import ESP32Reader

    HardwareFactory.register_driver("esp32", ESP32Reader)
except ImportError:
    pass

try:
    from .atheros import AtherosReader

    HardwareFactory.register_driver("atheros_ar9300", AtherosReader)
    HardwareFactory.register_driver("atheros", AtherosReader)  # Alternative name
except ImportError:
    pass

# Placeholder for future drivers
# try:
#     from .qualcomm import QualcommReader
#     HardwareFactory.register_driver('qualcomm', QualcommReader)
# except ImportError:
#     pass

# try:
#     from .broadcom import BroadcomReader
#     HardwareFactory.register_driver('broadcom', BroadcomReader)
# except ImportError:
#     pass

# try:
#     from .mediatek import MediatekReader
#     HardwareFactory.register_driver('mediatek', MediatekReader)
# except ImportError:
#     pass


def CSIReader(hardware_type: str, config: dict = None) -> CSIReaderBase:
    """
    Factory function to create CSI reader for specified hardware.

    Args:
        hardware_type: Type of hardware ('intel_5300', 'esp32', etc.)
        config: Hardware configuration dictionary

    Returns:
        Configured CSI reader instance

    Example:
        >>> reader = CSIReader('esp32', {
        ...     'sampling_rate': 100,
        ...     'channel': 6,
        ...     'bandwidth': 20
        ... })
    """
    # Convert dict config to HardwareConfig object
    if config is None:
        config = {}

    # Set default values
    default_config = {
        "sampling_rate": 100.0,
        "channel": 6,
        "bandwidth": 20.0,
        "antenna_config": [1],
        "calibration_required": True,
        "buffer_size": 1000,
        "timeout": 1.0,
    }

    # Merge with user config
    merged_config = {**default_config, **config}

    # Create HardwareConfig object
    hardware_config = HardwareConfig(
        sampling_rate=merged_config["sampling_rate"],
        channel=merged_config["channel"],
        bandwidth=merged_config["bandwidth"],
        antenna_config=merged_config["antenna_config"],
        calibration_required=merged_config["calibration_required"],
        buffer_size=merged_config["buffer_size"],
        timeout=merged_config["timeout"],
        additional_params=merged_config.get("additional_params"),
    )

    return HardwareFactory.create_reader(hardware_type, hardware_config)


def list_supported_hardware():
    """
    Get list of currently supported hardware platforms.

    Returns:
        List of supported hardware type strings
    """
    return HardwareFactory.list_supported_hardware()


def get_hardware_info(hardware_type: str) -> dict:
    """
    Get information about specific hardware platform.

    Args:
        hardware_type: Type of hardware to get info for

    Returns:
        Dictionary with hardware specifications
    """
    hardware_specs = {
        "intel_5300": {
            "name": "Intel WiFi Link 5300",
            "subcarriers": 30,
            "max_antennas": 3,
            "typical_sampling_rate": 1000,
            "bandwidth_options": [20, 40],
            "notes": "Research standard, requires modified driver",
        },
        "esp32": {
            "name": "ESP32 with CSI",
            "subcarriers": [64, 128],
            "max_antennas": 2,
            "typical_sampling_rate": [100, 250, 500],
            "bandwidth_options": [20, 40],
            "notes": "IoT applications, affordable hardware",
        },
        "atheros_ar9300": {
            "name": "Atheros AR9300",
            "subcarriers": 56,
            "max_antennas": 3,
            "typical_sampling_rate": 1000,
            "bandwidth_options": [20, 40],
            "notes": "Legacy research platform",
        },
    }

    return hardware_specs.get(hardware_type, {})


# Export public API
__all__ = [
    # Core classes
    "CSIData",
    "HardwareConfig",
    "CSIReaderBase",
    "HardwareFactory",
    # Factory functions
    "CSIReader",
    # Utility functions
    "list_supported_hardware",
    "get_hardware_info",
    "validate_csi_data",
    "normalize_csi_amplitude",
]

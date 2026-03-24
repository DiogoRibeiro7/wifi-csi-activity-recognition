"""
Base classes and interfaces for WiFi CSI hardware abstraction.

This module defines the abstract base classes that all hardware drivers must implement,
ensuring a consistent interface across different WiFi platforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class CSIData:
    """
    Standardized CSI data format across all hardware platforms.

    This class provides a unified representation of Channel State Information
    regardless of the underlying hardware platform.

    Attributes:
        timestamp: Unix timestamp when CSI was captured
        amplitude: CSI amplitude data [n_rx, n_tx, n_subcarriers]
        phase: CSI phase data [n_rx, n_tx, n_subcarriers]
        frequency: Center frequency in MHz
        bandwidth: Channel bandwidth in MHz
        n_tx: Number of transmit antennas
        n_rx: Number of receive antennas
        n_subcarriers: Number of subcarriers
        rssi: Received Signal Strength Indicator (optional)
        noise_floor: Noise floor measurement (optional)
        metadata: Hardware-specific additional information
    """

    timestamp: float
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: float
    bandwidth: float
    n_tx: int
    n_rx: int
    n_subcarriers: int
    rssi: Optional[float] = None
    noise_floor: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate CSI data after initialization."""
        expected_shape = (self.n_rx, self.n_tx, self.n_subcarriers)

        if self.amplitude.shape != expected_shape:
            raise ValueError(
                f"Amplitude shape {self.amplitude.shape} does not match "
                f"expected shape {expected_shape}"
            )

        if self.phase.shape != expected_shape:
            raise ValueError(
                f"Phase shape {self.phase.shape} does not match "
                f"expected shape {expected_shape}"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of CSI data as (n_rx, n_tx, n_subcarriers)."""
        return (self.n_rx, self.n_tx, self.n_subcarriers)

    @property
    def complex_csi(self) -> np.ndarray:
        """Get complex CSI representation (amplitude * e^(j*phase))."""
        return self.amplitude * np.exp(1j * self.phase)

    def to_dict(self) -> Dict[str, Any]:
        """Convert CSI data to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "amplitude": self.amplitude.tolist(),
            "phase": self.phase.tolist(),
            "frequency": self.frequency,
            "bandwidth": self.bandwidth,
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "rssi": self.rssi,
            "noise_floor": self.noise_floor,
            "metadata": self.metadata,
        }


@dataclass
class HardwareConfig:
    """
    Configuration for hardware-specific parameters.

    This class stores hardware-specific configuration parameters
    that may vary between different platforms.
    """

    sampling_rate: float  # Packets per second
    channel: int  # WiFi channel number
    bandwidth: float  # Channel bandwidth in MHz
    antenna_config: List[int]  # Available antenna indices
    calibration_required: bool = True
    buffer_size: int = 1000
    timeout: float = 1.0
    additional_params: Optional[Dict[str, Any]] = None


class CSIReaderBase(ABC):
    """
    Abstract base class for all CSI hardware readers.

    This class defines the interface that all hardware-specific implementations
    must follow to ensure consistent behavior across platforms.
    """

    def __init__(self, config: HardwareConfig):
        """
        Initialize the CSI reader with hardware configuration.

        Args:
            config: Hardware-specific configuration parameters
        """
        self.config = config
        self._is_connected = False
        self._is_streaming = False
        self._buffer: List[CSIData] = []

    @property
    def is_connected(self) -> bool:
        """Check if hardware is connected and ready."""
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        """Check if hardware is currently streaming data."""
        return self._is_streaming

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to hardware.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from hardware and cleanup resources."""
        pass

    @abstractmethod
    def start_streaming(self) -> None:
        """Start streaming CSI data from hardware."""
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop streaming CSI data."""
        pass

    @abstractmethod
    def read_packet(self) -> Optional[CSIData]:
        """
        Read a single CSI packet from hardware.

        Returns:
            CSIData object if packet available, None if no data
        """
        pass

    def read_batch(
        self, num_packets: int, timeout: Optional[float] = None
    ) -> List[CSIData]:
        """
        Read multiple CSI packets from hardware.

        Args:
            num_packets: Number of packets to read
            timeout: Maximum time to wait for packets (seconds)

        Returns:
            List of CSIData objects
        """
        packets = []
        timeout = timeout or self.config.timeout
        start_time = datetime.now()

        while len(packets) < num_packets:
            packet = self.read_packet()
            if packet is not None:
                packets.append(packet)

            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                break

        return packets

    def stream(self, buffer_size: Optional[int] = None) -> Iterator[CSIData]:
        """
        Create an iterator for continuous CSI data streaming.

        Args:
            buffer_size: Size of internal buffer for streaming

        Yields:
            CSIData objects as they become available
        """
        if not self.is_connected:
            raise RuntimeError("Hardware not connected")

        self.start_streaming()

        try:
            while self._is_streaming:
                packet = self.read_packet()
                if packet is not None:
                    yield packet
        finally:
            self.stop_streaming()

    @abstractmethod
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware-specific information.

        Returns:
            Dictionary containing hardware details
        """
        pass

    @abstractmethod
    def calibrate(self) -> bool:
        """
        Perform hardware-specific calibration.

        Returns:
            True if calibration successful, False otherwise
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class HardwareFactory:
    """Factory class for creating hardware-specific CSI readers."""

    _registered_drivers = {}

    @classmethod
    def register_driver(cls, hardware_type: str, driver_class: type):
        """
        Register a new hardware driver.

        Args:
            hardware_type: Unique identifier for hardware type
            driver_class: Class implementing CSIReaderBase
        """
        if not issubclass(driver_class, CSIReaderBase):
            raise ValueError("Driver must inherit from CSIReaderBase")

        cls._registered_drivers[hardware_type] = driver_class

    @classmethod
    def create_reader(cls, hardware_type: str, config: HardwareConfig) -> CSIReaderBase:
        """
        Create a CSI reader for specified hardware type.

        Args:
            hardware_type: Type of hardware to create reader for
            config: Hardware configuration

        Returns:
            Configured CSI reader instance

        Raises:
            ValueError: If hardware type not supported
        """
        if hardware_type not in cls._registered_drivers:
            raise ValueError(
                f"Unsupported hardware type: {hardware_type}. "
                f"Supported types: {list(cls._registered_drivers.keys())}"
            )

        driver_class = cls._registered_drivers[hardware_type]
        return driver_class(config)

    @classmethod
    def list_supported_hardware(cls) -> List[str]:
        """Get list of supported hardware types."""
        return list(cls._registered_drivers.keys())


# Utility functions
def validate_csi_data(csi_data: CSIData) -> bool:
    """
    Validate CSI data for consistency and reasonable values.

    Args:
        csi_data: CSI data to validate

    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check for NaN or infinite values
        if np.any(np.isnan(csi_data.amplitude)) or np.any(np.isinf(csi_data.amplitude)):
            return False

        if np.any(np.isnan(csi_data.phase)) or np.any(np.isinf(csi_data.phase)):
            return False

        # Check reasonable value ranges
        if csi_data.amplitude.min() < 0:
            return False

        # Phase should be between -π and π
        if csi_data.phase.min() < -np.pi or csi_data.phase.max() > np.pi:
            return False

        # Check timestamp is reasonable (not too old or in future)
        current_time = datetime.now().timestamp()
        if abs(csi_data.timestamp - current_time) > 86400:  # 24 hours
            return False

        return True

    except (AttributeError, ValueError, TypeError):
        return False


def normalize_csi_amplitude(
    amplitude: np.ndarray, method: str = "minmax"
) -> np.ndarray:
    """
    Normalize CSI amplitude data.

    Args:
        amplitude: Raw amplitude data
        method: Normalization method ("minmax", "zscore", "log")

    Returns:
        Normalized amplitude data
    """
    if method == "minmax":
        min_val, max_val = amplitude.min(), amplitude.max()
        if max_val > min_val:
            return (amplitude - min_val) / (max_val - min_val)
        else:
            return amplitude

    elif method == "zscore":
        mean_val, std_val = amplitude.mean(), amplitude.std()
        if std_val > 0:
            return (amplitude - mean_val) / std_val
        else:
            return amplitude - mean_val

    elif method == "log":
        return np.log(amplitude + 1e-8)  # Add small epsilon to avoid log(0)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

"""Intel 5300 WiFi hardware driver for CSI extraction.

This driver parses real CSI packets produced by the `linux-80211n-csitool`
project and converts them into the standardized
:class:`~wifi_activity_recognition.hardware.base.CSIData` format. It also keeps
the previous synthetic behaviour through a ``mock`` mode for environments where
the actual hardware or capture files are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import CSIData, CSIReaderBase, HardwareConfig, normalize_csi_amplitude

try:  # pragma: no cover - optional dependency
    import csiread  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    csiread = None  # type: ignore


@dataclass
class Intel5300State:
    """Internal state for the Intel 5300 driver."""

    calibrated: bool = False
    packet_index: int = 0


class Intel5300Reader(CSIReaderBase):
    """Driver for Intel 5300 CSI extraction."""

    def __init__(self, config: HardwareConfig):
        """Initialize the Intel 5300 reader.

        Parameters in ``config.additional_params``:

        ``mode``
            ``"file"`` to parse a ``.dat`` capture, ``"mock"`` for synthetic
            generation. Defaults to ``"mock"``.
        ``file_path``
            Path to the ``.dat`` capture when ``mode`` is ``"file"``.
        ``amplitude_method``
            Normalization method passed to
            :func:`~wifi_activity_recognition.hardware.base.normalize_csi_amplitude`.
        """
        super().__init__(config)
        params = config.additional_params or {}
        self.n_tx: int = int(params.get("n_tx", 3))
        self.n_rx: int = int(params.get("n_rx", len(config.antenna_config)))
        self.n_subcarriers: int = int(params.get("n_subcarriers", 30))
        self.mode: str = str(params.get("mode", "mock")).lower()
        self.file_path: Optional[Path] = (
            Path(params.get("file_path")) if params.get("file_path") else None
        )
        self.amplitude_method: str = str(params.get("amplitude_method", "minmax"))
        self.state = Intel5300State(calibrated=not config.calibration_required)
        self._intel: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------
    def connect(self) -> bool:  # type: ignore[override]
        """Establish connection to the Intel 5300 hardware or data source."""
        if self.mode == "file":
            if self.file_path is None or not self.file_path.exists():
                raise FileNotFoundError("CSI capture file not found")
            if csiread is None:  # pragma: no cover - handled in tests
                raise RuntimeError("csiread is required for file mode")
            self._intel = csiread.Intel(
                str(self.file_path), nrxnum=self.n_rx, ntxnum=self.n_tx
            )
            self._intel.read()
            self.state.packet_index = 0
        self._is_connected = True
        return True

    def disconnect(self) -> None:  # type: ignore[override]
        """Disconnect from hardware and reset state."""
        self._intel = None
        self._is_connected = False
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Streaming Control
    # ------------------------------------------------------------------
    def start_streaming(self) -> None:  # type: ignore[override]
        """Begin streaming CSI data."""
        if not self._is_connected:
            raise RuntimeError("Hardware not connected")
        self._is_streaming = True

    def stop_streaming(self) -> None:  # type: ignore[override]
        """Stop the CSI data stream."""
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Data Acquisition
    # ------------------------------------------------------------------
    def read_packet(self) -> Optional[CSIData]:  # type: ignore[override]
        """Read a CSI packet from the configured source."""
        if not (self._is_connected and self._is_streaming):
            return None

        if not self.state.calibrated:
            self.calibrate()

        if self.mode == "file" and self._intel is not None:
            if self.state.packet_index >= self._intel.count:
                return None
            raw_csi = self._intel.csi[self.state.packet_index]
            # csiread outputs [subcarrier, Nrx, Ntx]
            csi = np.transpose(raw_csi, (1, 2, 0))
            amplitude = normalize_csi_amplitude(
                np.abs(csi), method=self.amplitude_method
            )
            phase = self._calibrate_phase(np.angle(csi))
            timestamp = float(self._intel.timestamp_low[self.state.packet_index])
            self.state.packet_index += 1
        else:
            amplitude = np.abs(
                np.random.randn(self.n_rx, self.n_tx, self.n_subcarriers)
            )
            phase = np.random.uniform(
                -np.pi, np.pi, size=(self.n_rx, self.n_tx, self.n_subcarriers)
            )
            timestamp = datetime.now().timestamp()

        frequency = 2412 + (self.config.channel - 1) * 5
        bandwidth = self.config.bandwidth
        csi = CSIData(
            timestamp=timestamp,
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth,
            n_tx=self.n_tx,
            n_rx=self.n_rx,
            n_subcarriers=self.n_subcarriers,
            metadata={"hardware": "Intel 5300"},
        )

        self._buffer.append(csi)
        if len(self._buffer) > self.config.buffer_size:
            self._buffer.pop(0)
        return csi

    # ------------------------------------------------------------------
    # Calibration & Info
    # ------------------------------------------------------------------
    def get_hardware_info(self) -> Dict[str, Any]:  # type: ignore[override]
        """Return basic hardware specifications."""
        return {
            "name": "Intel WiFi Link 5300",
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "supports_calibration": True,
        }

    def calibrate(self) -> bool:  # type: ignore[override]
        """Perform phase calibration."""
        self.state.calibrated = True
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _calibrate_phase(self, phase: np.ndarray) -> np.ndarray:
        """Remove linear phase trend and wrap to ``[-π, π]``."""
        unwrapped = np.unwrap(phase, axis=-1)
        k = np.arange(self.n_subcarriers)
        slope = (unwrapped[..., -1] - unwrapped[..., 0]) / (self.n_subcarriers - 1)
        intercept = unwrapped[..., 0]
        trend = slope[..., None] * k + intercept[..., None]
        calibrated = unwrapped - trend
        return ((calibrated + np.pi) % (2 * np.pi)) - np.pi


__all__ = ["Intel5300Reader"]

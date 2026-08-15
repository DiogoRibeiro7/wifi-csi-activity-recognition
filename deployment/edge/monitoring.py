"""Monitoring utilities for edge devices."""

from __future__ import annotations

import logging
from typing import Dict, Optional

try:  # pragma: no cover - psutil may not be installed in some envs
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

logger = logging.getLogger(__name__)


class EdgeMonitor:
    """Collect basic system metrics for diagnostics."""

    def get_memory_usage(self) -> float:
        """Return current process memory usage in MB."""
        if psutil is None:  # pragma: no cover
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)

    def get_cpu_usage(self) -> float:
        """Return system-wide CPU utilization percentage."""
        if psutil is None:  # pragma: no cover
            return 0.0
        return psutil.cpu_percent(interval=0.1)

    def get_temperature(self) -> Optional[float]:
        """Return CPU temperature if available."""
        if psutil is None:  # pragma: no cover
            return None
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            first = next(iter(temps.values()))
            return first[0].current
        except Exception:  # pragma: no cover - platform dependent
            return None

    def get_battery(self) -> Optional[float]:
        """Return battery percentage if available."""
        if psutil is None:  # pragma: no cover
            return None
        try:
            batt = psutil.sensors_battery()
            return batt.percent if batt else None
        except Exception:  # pragma: no cover - not all systems support
            return None

    def gather(self) -> Dict[str, Optional[float]]:
        """Gather all metrics in a dictionary."""
        metrics: Dict[str, Optional[float]] = {
            "memory_mb": self.get_memory_usage(),
            "cpu_percent": self.get_cpu_usage(),
            "temperature_c": self.get_temperature(),
            "battery_percent": self.get_battery(),
        }
        logger.debug("Edge metrics: %s", metrics)
        return metrics


__all__ = ["EdgeMonitor"]

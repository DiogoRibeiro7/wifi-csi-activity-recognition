"""Lightweight inference runtime for edge devices."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .device_profiles import DeviceProfile
from .optimization import quantize_dynamic


class EdgeRuntime:
    """Minimal inference runtime with optional quantization."""

    def __init__(self, model: torch.nn.Module, profile: DeviceProfile) -> None:
        """Initialize the runtime with model and device profile."""
        self.model = model
        self.profile = profile
        self.device = torch.device("cpu")
        if profile.quantization == "int8":
            self.model = quantize_dynamic(self.model, dtype=torch.qint8)
        elif profile.quantization == "fp16":
            self.model = quantize_dynamic(self.model, dtype=torch.float16)
        self.model.eval()

    def load_jit(self, path: Path) -> None:
        """Load a TorchScript model."""
        self.model = torch.jit.load(str(path))
        self.model.eval()

    @torch.no_grad()
    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run inference on ``tensor`` and return the output."""
        return self.model(tensor.to(self.device))

    def update_model(self, new_model: torch.nn.Module) -> None:
        """Replace the current model at runtime."""
        self.model = new_model.eval()

    def sync_metrics(self, endpoint: Optional[str] = None) -> None:
        """Synchronize metrics with a cloud endpoint (placeholder)."""
        _ = endpoint  # In real implementations, send metrics to the cloud.


__all__ = ["EdgeRuntime"]

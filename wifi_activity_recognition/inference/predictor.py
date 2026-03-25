"""Model-based activity recognition utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from ..hardware.base import CSIData
from ..models import load_model


class ActivityRecognizer:
    """Run single-packet activity predictions using a trained model.

    Parameters
    ----------
    model:
        Either a :class:`torch.nn.Module` instance or path to a saved model.
    class_names:
        Optional mapping of class indices to human-readable labels.
    device:
        Torch device to run inference on. Defaults to CUDA when available.
    """

    def __init__(
        self,
        model: Union[str, Path, nn.Module],
        class_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the recognizer with a model or model path."""
        if isinstance(model, (str, Path)):
            self.model = load_model(model, map_location="cpu")
        else:
            self.model = model
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        if class_names is None:
            num_classes = getattr(self.model, "classifier", None)
            if isinstance(num_classes, nn.Linear):
                n = num_classes.out_features
            else:
                n = 0
            class_names = [str(i) for i in range(n)]
        self.class_names: List[str] = list(class_names)

    def _to_tensor(self, csi_data: CSIData) -> torch.Tensor:
        """Convert :class:`CSIData` into model input tensor."""
        amp = np.transpose(csi_data.amplitude, (2, 0, 1))
        amp = amp.reshape(csi_data.n_subcarriers, csi_data.n_rx * csi_data.n_tx)
        if amp.shape[1] < 8:
            reps = int(np.ceil(8 / amp.shape[1]))
            amp = np.tile(amp, (1, reps))[:, :8]
        tensor = torch.tensor(amp, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, csi_data: CSIData) -> Tuple[str, float]:
        """Predict activity from a single CSI packet."""
        x = self._to_tensor(csi_data)
        logits = self.model(x)[0]
        probs = torch.softmax(logits, dim=0)
        conf, idx = torch.max(probs, dim=0)
        label = self.class_names[int(idx)] if self.class_names else str(int(idx))
        return label, float(conf)

"""Domain adaptation utilities for CSI-based activity models."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..hardware.base import CSIData

try:  # pragma: no cover - optional dependency
    from ..models.base import BaseActivityModel  # type: ignore
except Exception:  # pragma: no cover
    BaseActivityModel = nn.Module  # type: ignore


class DomainAdapter:
    """Adapt pre-trained models to new target environments.

    This minimalist implementation performs mean-centering of target domain
    features (similar in spirit to CORAL) to mitigate distribution shift.
    """

    def __init__(self, source_model: BaseActivityModel) -> None:
        """Initialize the adapter with a source-domain model."""
        self.model = source_model
        self._mean: Optional[np.ndarray] = None

    def _extract_feature(self, csi: CSIData) -> np.ndarray:
        """Extract a simple feature vector from :class:`CSIData`."""
        return np.array([float(csi.amplitude.mean())], dtype=np.float32)

    def adapt_to_target(
        self, target_csi: Iterable[CSIData], method: str = "coral"
    ) -> None:
        """Estimate target-domain statistics for adaptation.

        Parameters
        ----------
        target_csi:
            Iterable of unlabeled :class:`CSIData` from the target domain.
        method:
            Currently only ``"coral"`` (mean-centering) is supported.
        """
        if method != "coral":  # pragma: no cover - defensive programming
            raise ValueError(f"Unsupported adaptation method: {method}")

        feats = [self._extract_feature(csi) for csi in target_csi]
        if not feats:
            raise ValueError("No target CSI data provided for adaptation")
        self._mean = np.mean(np.vstack(feats), axis=0)

    def predict(self, csi: CSIData) -> Tuple[int, float]:
        """Predict activity label and confidence for a CSI packet."""
        feat = self._extract_feature(csi)
        if self._mean is not None:
            feat = feat - self._mean
        tensor = torch.tensor(feat, dtype=torch.float32)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=0)
        conf, idx = torch.max(probs, dim=0)
        return int(idx), float(conf)

    def evaluate_adaptation(
        self, target_test: Sequence[Tuple[CSIData, int]]
    ) -> Dict[str, float]:
        """Evaluate adapted model on labeled target-domain data."""
        if not target_test:
            raise ValueError("Empty evaluation dataset")
        correct = 0
        for csi, label in target_test:
            pred, _ = self.predict(csi)
            correct += int(pred == label)
        accuracy = correct / len(target_test)
        return {"accuracy": accuracy}

"""Few-shot learning utilities for rapid activity adaptation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn

from ..hardware.base import CSIData

try:  # pragma: no cover
    from ..models.base import BaseActivityModel  # type: ignore
except Exception:  # pragma: no cover
    BaseActivityModel = nn.Module  # type: ignore


class FewShotLearner:
    """Meta-learning helper using prototype-based classification."""

    def __init__(
        self, base_model: BaseActivityModel, novelty_threshold: float = 5.0
    ) -> None:
        """Initialize the learner with a base model and novelty threshold."""
        self.model = base_model
        self.prototypes: Dict[str, np.ndarray] = {}
        self.novelty_threshold = float(novelty_threshold)

    def _embed(self, csi: CSIData) -> np.ndarray:
        feat = np.array([float(csi.amplitude.mean())], dtype=np.float32)
        tensor = torch.tensor(feat, dtype=torch.float32)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.detach().cpu().numpy().reshape(-1)

    def learn_new_activity(
        self, support_set: Iterable[CSIData], activity_name: str
    ) -> None:
        """Create prototype from a small support set for a new activity."""
        feats = [self._embed(csi) for csi in support_set]
        if not feats:
            raise ValueError("Support set is empty")
        self.prototypes[activity_name] = np.mean(np.vstack(feats), axis=0)

    def predict_with_confidence(self, csi_data: CSIData) -> Tuple[str, float, bool]:
        """Predict activity with novelty detection."""
        feat = self._embed(csi_data)
        if not self.prototypes:
            return "unknown", 0.0, True
        dists = {
            name: float(np.linalg.norm(feat - proto))
            for name, proto in self.prototypes.items()
        }
        best_name, best_dist = min(dists.items(), key=lambda kv: kv[1])
        confidence = float(np.exp(-best_dist))
        is_novel = best_dist > self.novelty_threshold
        return best_name, confidence, is_novel

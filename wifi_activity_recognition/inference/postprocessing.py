"""Post-processing helpers for inference outputs."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def smooth_probabilities(probs: Iterable[np.ndarray]) -> np.ndarray:
    """Average a sequence of probability vectors."""
    stacked = np.vstack(list(probs))
    return stacked.mean(axis=0)


def apply_confidence_threshold(
    confidence: float, label: str, threshold: float
) -> Optional[Tuple[str, float]]:
    """Return label when confidence exceeds threshold."""
    if confidence >= threshold:
        return label, confidence
    return None

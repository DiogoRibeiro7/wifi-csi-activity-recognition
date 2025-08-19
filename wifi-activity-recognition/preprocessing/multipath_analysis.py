"""Multipath component separation utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from ..hardware.base import CSIData


def separate_multipath_components(
    csi: CSIData,
    n_components: int = 3,
) -> List[CSIData]:
    """Decompose CSI into dominant multipath components using SVD.

    Parameters
    ----------
    n_components:
        Number of components to extract. Typical values are between 1 and 5.
    """
    if n_components <= 0 or n_components > csi.n_subcarriers:
        raise ValueError("invalid number of components")
    complex_csi = csi.complex_csi
    flat = complex_csi.reshape(csi.n_rx * csi.n_tx, csi.n_subcarriers)
    u, s, vh = np.linalg.svd(flat, full_matrices=False)
    comps = []
    for i in range(n_components):
        comp_flat = np.outer(u[:, i], vh[i, :]) * s[i]
        comp = comp_flat.reshape(csi.n_rx, csi.n_tx, csi.n_subcarriers)
        new_csi = replace(
            csi,
            amplitude=np.abs(comp),
            phase=np.angle(comp),
        )
        comps.append(new_csi)
    return comps


__all__ = ["separate_multipath_components"]

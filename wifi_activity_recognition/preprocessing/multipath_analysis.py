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

    The complex CSI matrix is reshaped to ``(n_rx * n_tx, n_subcarriers)`` and a
    singular value decomposition is performed.  The ``n_components`` largest
    singular vectors are then reconstructed and converted back into
    :class:`CSIData` objects containing the corresponding amplitude and phase.

    Parameters
    ----------
    csi:
        Input CSI sample.
    n_components:
        Number of components to extract. Typical values range from 1 to 5.

    Returns
    -------
    list of :class:`CSIData`
        ``n_components`` dominant multipath components ordered by energy.

    Raises
    ------
    ValueError
        If ``n_components`` is not in ``[1, min(n_rx * n_tx, n_subcarriers)]``.
    """
    max_rank = min(csi.n_rx * csi.n_tx, csi.n_subcarriers)
    if n_components <= 0 or n_components > max_rank:
        raise ValueError("invalid number of components")
    complex_csi = csi.complex_csi
    flat = complex_csi.reshape(csi.n_rx * csi.n_tx, csi.n_subcarriers)
    u, s, vh = np.linalg.svd(flat, full_matrices=False)
    comps: List[CSIData] = []
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

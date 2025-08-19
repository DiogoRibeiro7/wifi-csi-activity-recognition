"""Differential privacy utilities for federated learning."""

from __future__ import annotations

from typing import Dict

import torch

StateDict = Dict[str, torch.Tensor]


def clip_gradients(state: StateDict, max_norm: float) -> StateDict:
    """Clip the L2 norm of parameters in ``state`` to ``max_norm``."""
    total_norm = torch.sqrt(sum(t.pow(2).sum() for t in state.values()))
    if total_norm <= max_norm:
        return state
    scale = max_norm / (total_norm + 1e-6)
    return {k: v * scale for k, v in state.items()}


def add_gaussian_noise(state: StateDict, std: float) -> StateDict:
    """Add Gaussian noise with standard deviation ``std`` to ``state``."""
    return {k: v + torch.randn_like(v) * std for k, v in state.items()}


def secure_aggregate(states: Dict[str, StateDict]) -> StateDict:
    """Securely aggregate masked updates.

    For the purposes of this library we implement a simple summation which can
    be replaced by an actual secure aggregation protocol. The ``states`` dict is
    expected to contain unmasked updates.
    """
    aggregated: StateDict = {}
    for key in next(iter(states.values())).keys():
        aggregated[key] = sum(s[key] for s in states.values())
    return aggregated


__all__ = ["clip_gradients", "add_gaussian_noise", "secure_aggregate"]

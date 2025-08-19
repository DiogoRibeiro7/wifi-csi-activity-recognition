"""Federated server coordinating client training."""

from __future__ import annotations

import copy
import random
import typing as t
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Sequence

import torch
from torch import nn

from .aggregators import fed_avg

if t.TYPE_CHECKING:  # pragma: no cover
    from .client import FederatedClient

StateDict = Dict[str, torch.Tensor]


@dataclass
class FederatedServer:
    """Server orchestrating federated learning rounds."""

    model: nn.Module
    clients: Sequence["FederatedClient"]
    aggregator: Callable[[Iterable[tuple[StateDict, int]]], StateDict] = fed_avg
    fraction: float = 1.0
    rng: random.Random = field(default_factory=random.Random)

    def _select_clients(self) -> List["FederatedClient"]:
        """Select a subset of clients according to ``fraction``."""
        num_clients = max(1, int(len(self.clients) * self.fraction))
        return self.rng.sample(list(self.clients), num_clients)

    def train_round(self, epochs: int) -> None:
        """Run a single federated learning round."""
        selected = self._select_clients()
        global_state = self.model.state_dict()
        updates: List[tuple[StateDict, int]] = []
        for client in selected:
            client.update_model(copy.deepcopy(global_state))
            state, n_samples = client.train(epochs)
            updates.append((state, n_samples))
        new_state = self.aggregator(updates)
        self.model.load_state_dict(new_state)


__all__ = ["FederatedServer"]

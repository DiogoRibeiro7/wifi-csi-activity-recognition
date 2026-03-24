"""Federated learning client implementation."""

from __future__ import annotations

import copy
import typing as t
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from ..trainer import Trainer
from .privacy import add_gaussian_noise, clip_gradients

if t.TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ...datasets import Dataset

StateDict = Dict[str, torch.Tensor]


@dataclass
class FederatedClient:
    """Client participating in federated learning rounds.

    Parameters
    ----------
    model:
        Local model instance to be trained on the client's data.
    dataset:
        Local dataset split. Raw CSI never leaves the client.
    trainer_kwargs:
        Additional arguments forwarded to
        :class:`~wifi_activity_recognition.training.Trainer`.
    noise_std:
        Standard deviation of Gaussian noise added for differential privacy.
    clip_norm:
        Gradient clipping norm used before noise injection.
    hardware:
        Identifier of the underlying hardware (e.g., ``"intel5300"``).
    environment:
        Label of the deployment environment (e.g., ``"home"`` or ``"office"``).
    """

    model: nn.Module
    dataset: "Dataset"
    trainer_kwargs: Dict[str, object] = field(default_factory=dict)
    noise_std: float = 0.0
    clip_norm: Optional[float] = None
    hardware: str = "generic"
    environment: str = "default"

    def update_model(self, state: StateDict) -> None:
        """Synchronize local model with the global ``state``."""
        self.model.load_state_dict(state)

    def train(self, epochs: int) -> tuple[StateDict, int]:
        """Run local training and return updated weights and sample count."""
        trainer = Trainer(
            model=copy.deepcopy(self.model), dataset=self.dataset, **self.trainer_kwargs
        )
        trainer.train(epochs)
        state = trainer.model.state_dict()
        if self.clip_norm is not None:
            state = clip_gradients(state, self.clip_norm)
        if self.noise_std > 0:
            state = add_gaussian_noise(state, self.noise_std)
        num_samples = len(self.dataset.train[0])
        return state, num_samples


__all__ = ["FederatedClient"]

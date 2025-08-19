"""Training callbacks for monitoring and control."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

if t.TYPE_CHECKING:  # pragma: no cover
    from .trainer import Trainer


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:  # pragma: no cover - hooks
        """Run at the beginning of training."""

    def on_epoch_begin(
        self, trainer: "Trainer", epoch: int
    ) -> None:  # pragma: no cover - hooks
        """Run at the start of each epoch."""

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:  # pragma: no cover - hooks
        """Run after each epoch with training logs."""

    def on_train_end(self, trainer: "Trainer") -> None:  # pragma: no cover - hooks
        """Run after training has finished."""


@dataclass
class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving."""

    patience: int = 5
    monitor: str = "val_loss"
    min_delta: float = 0.0
    _best: Optional[float] = None
    _num_bad_epochs: int = 0

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Check metric value and request training stop if patience exceeded."""
        current = logs.get(self.monitor)
        if current is None:
            return
        if self._best is None or current < self._best - self.min_delta:
            self._best = current
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self.patience:
                trainer.stop_training = True


@dataclass
class ModelCheckpoint(Callback):
    """Save the model when a monitored metric improves."""

    filepath: Path
    monitor: str = "val_loss"
    save_best_only: bool = True
    _best: Optional[float] = None

    def __post_init__(self) -> None:
        """Ensure the checkpoint directory exists."""
        self.filepath = Path(self.filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Save model state if monitored metric improves."""
        current = logs.get(self.monitor)
        if current is None:
            return
        if not self.save_best_only or self._best is None or current < self._best:
            self._best = current
            state_dict = (
                trainer.model.module.state_dict()
                if isinstance(trainer.model, torch.nn.DataParallel)
                else trainer.model.state_dict()
            )
            torch.save(state_dict, self.filepath)


__all__ = ["Callback", "EarlyStopping", "ModelCheckpoint"]

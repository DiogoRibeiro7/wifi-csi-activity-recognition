"""Training callbacks for monitoring and control."""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from ..models import save_model_artifact

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
            save_model_artifact(
                trainer.model,
                self.filepath,
                metadata={"checkpoint_monitor": self.monitor, "checkpoint_value": current},
            )


@dataclass
class LRScheduler(Callback):
    """Adjust the learning rate according to a ``torch`` scheduler."""

    scheduler: (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    monitor: str = "val_loss"

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Step the learning-rate scheduler."""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()


try:  # pragma: no cover - tensorboard is optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - runtime import guard
    SummaryWriter = None


@dataclass
class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    log_dir: Path

    def __post_init__(self) -> None:
        """Create the summary writer."""
        if SummaryWriter is None:  # pragma: no cover - defensive
            raise ImportError("tensorboard is required for TensorBoardLogger")
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, float]
    ) -> None:
        """Write scalar metrics for the current epoch."""
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Close the summary writer."""
        self.writer.close()


__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRScheduler",
    "TensorBoardLogger",
]

"""Tests for training callbacks."""

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.training import (  # type: ignore  # noqa: E402
    EarlyStopping,
    LRScheduler,
    ModelCheckpoint,
    TensorBoardLogger,
)

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    TENSORBOARD_AVAILABLE = False


class DummyTrainer:
    """Minimal trainer stub for callback tests."""

    def __init__(self):
        """Initialize the dummy trainer with a simple model."""
        self.stop_training = False
        self.model = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)


def test_early_stopping_triggers():
    """Early stopping should flag ``stop_training`` after patience epochs."""
    trainer = DummyTrainer()
    cb = EarlyStopping(patience=1)
    cb.on_epoch_end(trainer, 1, {"val_loss": 1.0})
    cb.on_epoch_end(trainer, 2, {"val_loss": 1.0})
    assert trainer.stop_training is True


def test_model_checkpoint_saves(tmp_path: Path):
    """Model checkpoint callback should write model state to disk."""
    trainer = DummyTrainer()
    ckpt = tmp_path / "best.pt"
    cb = ModelCheckpoint(ckpt)
    cb.on_epoch_end(trainer, 1, {"val_loss": 1.0})
    assert ckpt.exists()


def test_lr_scheduler_steps():
    """Learning rate scheduler callback should step each epoch."""
    trainer = DummyTrainer()
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
    cb = LRScheduler(scheduler)
    cb.on_epoch_end(trainer, 1, {"val_loss": 1.0})
    assert scheduler.last_epoch == 1


@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_tensorboard_logger_writes(tmp_path: Path):
    """Tensorboard logger should create event files."""
    logger = TensorBoardLogger(tmp_path)
    logger.on_epoch_end(DummyTrainer(), 1, {"val_loss": 1.0})
    logger.on_train_end(DummyTrainer())
    assert any(tmp_path.iterdir())

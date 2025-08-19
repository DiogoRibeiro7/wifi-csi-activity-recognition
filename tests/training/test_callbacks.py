"""Tests for training callbacks."""

import sys
import types
from pathlib import Path

from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.training import (  # type: ignore  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
)


class DummyTrainer:
    """Minimal trainer stub for callback tests."""

    def __init__(self):
        """Initialize the dummy trainer with a simple model."""
        self.stop_training = False
        self.model = nn.Linear(1, 1)


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

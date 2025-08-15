"""Tests for the Trainer class."""

import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    Dataset,
    split_dataset,
)
from wifi_activity_recognition.training import Trainer  # type: ignore  # noqa: E402


def make_dummy_dataset() -> Dataset:
    data = np.random.rand(30, 1, 8, 8).astype(np.float32)
    labels = np.random.randint(0, 2, 30)
    train, val, test = split_dataset(data, labels, val_ratio=0.2, test_ratio=0.2)
    return Dataset(train=train, val=val, test=test)


def test_trainer_train_loop(tmp_path: Path):
    dataset = make_dummy_dataset()
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 8 * 8, 2))
    trainer = Trainer(model=model, dataset=dataset, batch_size=8, learning_rate=1e-2)
    trainer.train(epochs=2)
    metrics = trainer.get_metrics()
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["val_accuracy"] <= 1.0
    out_path = tmp_path / "model.pt"
    trainer.save_model(out_path)
    assert out_path.exists()

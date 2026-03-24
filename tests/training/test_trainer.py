"""Tests for the Trainer class."""

from pathlib import Path

import numpy as np
import pytest
from torch import nn

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    TENSORBOARD_AVAILABLE = False


from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    Dataset,
    split_dataset,
)
from wifi_activity_recognition.training import Trainer  # type: ignore  # noqa: E402


def make_dummy_dataset() -> Dataset:
    """Create a small random dataset for training tests."""
    data = np.random.rand(30, 1, 8, 8).astype(np.float32)
    labels = np.random.randint(0, 2, 30)
    train, val, test = split_dataset(data, labels, val_ratio=0.2, test_ratio=0.2)
    return Dataset(train=train, val=val, test=test)


@pytest.mark.skipif(not TENSORBOARD_AVAILABLE, reason="tensorboard not installed")
def test_trainer_train_loop(tmp_path: Path):
    """Train a simple model and ensure metrics and checkpoint are produced."""
    dataset = make_dummy_dataset()
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 8 * 8, 2))
    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=8,
        learning_rate=1e-2,
        tensorboard_log_dir=tmp_path,
        use_lr_scheduler=True,
    )
    trainer.train(epochs=2)
    metrics = trainer.get_metrics()
    for key in [
        "train_accuracy",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
    ]:
        assert 0.0 <= metrics[key] <= 1.0
    assert len(metrics["val_per_class_accuracy"]) == 2
    assert np.array(metrics["val_confusion_matrix"]).shape == (2, 2)
    out_path = tmp_path / "model.pt"
    trainer.save_model(out_path)
    assert out_path.exists()
    assert any(tmp_path.iterdir())  # tensorboard files


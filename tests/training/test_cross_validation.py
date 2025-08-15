import sys
import types
from pathlib import Path

import numpy as np
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


def make_dataset() -> Dataset:
    data = np.random.rand(20, 1, 4, 4).astype(np.float32)
    labels = np.random.randint(0, 2, 20)
    train, val, test = split_dataset(data, labels, val_ratio=0.2, test_ratio=0.2)
    return Dataset(train=train, val=val, test=test)


def test_cross_validation_runs():
    dataset = make_dataset()
    model = nn.Sequential(nn.Flatten(), nn.Linear(1 * 4 * 4, 2))
    trainer = Trainer(model=model, dataset=dataset, batch_size=4, learning_rate=1e-2)
    results = trainer.cross_validate(folds=3, epochs=1)
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in results
        assert 0.0 <= results[key] <= 1.0

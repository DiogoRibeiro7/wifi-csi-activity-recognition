"""Functional tests for user-facing CLI workflows."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli

pytestmark = [pytest.mark.functional]


def test_evaluate_writes_report_and_prints_metrics(monkeypatch, tmp_path: Path) -> None:
    """Evaluate should orchestrate model loading, dataset loading, and report saving."""
    calls: dict[str, object] = {}

    class FakeDataset:
        def __init__(self) -> None:
            self.test = ([0, 1, 2], [0, 1, 0])

        @classmethod
        def from_files(cls, data_path, labels_path, hardware_type=None, **_kwargs):
            calls["data_path"] = str(data_path)
            calls["labels_path"] = str(labels_path)
            calls["hardware_type"] = hardware_type
            return cls()

    class FakeTrainer:
        def __init__(self, model, dataset) -> None:
            calls["trainer_model"] = model
            calls["trainer_dataset"] = dataset

        def evaluate(self, split="test"):
            calls["split"] = split
            return {
                "accuracy": 0.91,
                "precision": 0.87,
                "recall": 0.85,
                "f1_score": 0.86,
            }

    fake_datasets = types.ModuleType("wifi_activity_recognition.datasets")
    fake_datasets.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.datasets", fake_datasets)

    fake_models = types.ModuleType("wifi_activity_recognition.models")
    fake_models.load_model = lambda path: {"model_path": str(path)}
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.models", fake_models)

    fake_training = types.ModuleType("wifi_activity_recognition.training")
    fake_training.Trainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.training", fake_training)

    def fake_save_results(results, path) -> None:
        Path(path).write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.save_evaluation_results",
        fake_save_results,
    )

    model_path = tmp_path / "model.pt"
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    output_path = tmp_path / "evaluation.json"
    model_path.write_text("model", encoding="utf-8")
    data_path.write_text("data", encoding="utf-8")
    labels_path.write_text("labels", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--labels",
            str(labels_path),
            "--hardware",
            "esp32",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["data_path"] == str(data_path)
    assert calls["labels_path"] == str(labels_path)
    assert calls["hardware_type"] == "esp32"
    assert calls["split"] == "test"
    assert "Accuracy: 0.910" in result.output
    assert output_path.exists()

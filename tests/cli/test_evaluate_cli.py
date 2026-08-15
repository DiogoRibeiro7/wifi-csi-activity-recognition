"""CLI integration tests for evaluation workflows."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from click.testing import CliRunner

from wifi_activity_recognition.cli import cli


def test_evaluate_command_uses_dataset_and_trainer_contract(
    monkeypatch, tmp_path: Path
) -> None:
    """The evaluate command should use explicit data and labels inputs."""
    calls: dict[str, object] = {}

    class _FakeDataset:
        def __init__(self) -> None:
            self.test = ([0, 1, 2], [0, 1, 0])

        @classmethod
        def from_files(cls, data_path, labels_path, hardware_type=None, **_kwargs):
            calls["data_path"] = str(data_path)
            calls["labels_path"] = str(labels_path)
            calls["hardware_type"] = hardware_type
            return cls()

    class _FakeTrainer:
        def __init__(self, model, dataset) -> None:
            calls["trainer_model"] = model
            calls["trainer_dataset"] = dataset

        def evaluate(self, split="test"):
            calls["split"] = split
            return {
                "accuracy": 0.9,
                "precision": 0.8,
                "recall": 0.85,
                "f1_score": 0.82,
            }

    fake_datasets = types.ModuleType("wifi_activity_recognition.datasets")
    fake_datasets.Dataset = _FakeDataset
    monkeypatch.setitem(
        sys.modules, "wifi_activity_recognition.datasets", fake_datasets
    )

    fake_models = types.ModuleType("wifi_activity_recognition.models")
    fake_models.load_model = lambda path: {"model_path": str(path)}
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.models", fake_models)

    fake_training = types.ModuleType("wifi_activity_recognition.training")
    fake_training.Trainer = _FakeTrainer
    monkeypatch.setitem(
        sys.modules, "wifi_activity_recognition.training", fake_training
    )

    def fake_save_results(results, path) -> None:
        Path(path).write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.save_evaluation_results",
        fake_save_results,
    )

    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")
    data_path = tmp_path / "data.npy"
    data_path.write_text("data", encoding="utf-8")
    labels_path = tmp_path / "labels.npy"
    labels_path.write_text("labels", encoding="utf-8")
    output_path = tmp_path / "evaluation.json"

    runner = CliRunner()
    result = runner.invoke(
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
    assert "Accuracy: 0.900" in result.output
    assert output_path.exists()

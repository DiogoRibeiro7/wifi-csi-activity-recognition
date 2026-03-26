"""CLI integration tests for prediction workflows."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli
from wifi_activity_recognition.hardware.base import CSIData


class _DummyRecognizer:
    """Minimal recognizer stub for CLI tests."""

    def __init__(self, model) -> None:
        """Store the model reference passed by the CLI."""
        self.model = model

    def predict(self, _sample: CSIData) -> tuple[str, float]:
        """Return a stable synthetic prediction."""
        return "walking", 0.95


def _make_csi_packet() -> CSIData:
    """Create a synthetic CSI packet for prediction tests."""
    amplitude = np.ones((1, 1, 4))
    phase = np.zeros((1, 1, 4))
    return CSIData(
        timestamp=0.0,
        amplitude=amplitude,
        phase=phase,
        frequency=2400.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=4,
    )


def test_predict_command_uses_loaded_csi_packets(monkeypatch, tmp_path: Path) -> None:
    """The predict command should work with serialized CSIData packets."""
    fake_inference = types.ModuleType("wifi_activity_recognition.inference")
    fake_inference.ActivityRecognizer = _DummyRecognizer
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.inference", fake_inference)
    fake_models = types.ModuleType("wifi_activity_recognition.models")
    fake_models.load_model = lambda _path: object()
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.models", fake_models)
    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.load_csi_data",
        lambda _path: [_make_csi_packet(), _make_csi_packet()],
    )

    saved = {}

    def fake_save_predictions(predictions, confidences, path) -> None:
        saved["predictions"] = list(predictions)
        saved["confidences"] = list(confidences)
        Path(path).write_text("saved", encoding="utf-8")

    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.save_predictions",
        fake_save_predictions,
    )

    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "predictions.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--hardware",
            "esp32",
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert saved["predictions"] == ["walking", "walking"]
    assert saved["confidences"] == [0.95, 0.95]
    assert output_path.read_text(encoding="utf-8") == "saved"


def test_predict_command_rejects_non_csidata_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    """The predict command should fail clearly on array-only input files."""
    fake_inference = types.ModuleType("wifi_activity_recognition.inference")
    fake_inference.ActivityRecognizer = _DummyRecognizer
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.inference", fake_inference)
    fake_models = types.ModuleType("wifi_activity_recognition.models")
    fake_models.load_model = lambda _path: object()
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.models", fake_models)
    monkeypatch.setattr(
        "wifi_activity_recognition.utils.io.load_csi_data",
        lambda _path: np.ones((2, 2)),
    )

    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")
    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--hardware",
            "esp32",
            "--model",
            str(model_path),
            "--input",
            str(input_path),
        ],
    )

    assert result.exit_code != 0
    assert "serialized CSIData packets" in result.output

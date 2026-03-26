"""End-to-end style tests spanning multiple package layers."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli
from wifi_activity_recognition.hardware.base import CSIData

pytestmark = [pytest.mark.e2e, pytest.mark.smoke]


class _RoundTripReader:
    """Synthetic hardware reader for collection tests."""

    def __init__(self) -> None:
        self._packets = iter(
            [
                CSIData(
                    timestamp=0.0,
                    amplitude=np.ones((1, 1, 4)),
                    phase=np.zeros((1, 1, 4)),
                    frequency=2400.0,
                    bandwidth=20.0,
                    n_tx=1,
                    n_rx=1,
                    n_subcarriers=4,
                ),
                CSIData(
                    timestamp=1.0,
                    amplitude=np.ones((1, 1, 4)) * 2,
                    phase=np.zeros((1, 1, 4)),
                    frequency=2400.0,
                    bandwidth=20.0,
                    n_tx=1,
                    n_rx=1,
                    n_subcarriers=4,
                ),
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self):
        return self._packets


class _RoundTripRecognizer:
    """Prediction stub for end-to-end CLI roundtrips."""

    def __init__(self, _model) -> None:
        """Initialize the recognizer."""

    def predict(self, sample: CSIData) -> tuple[str, float]:
        """Generate deterministic labels from packet amplitude."""
        return ("walking" if float(sample.amplitude.mean()) < 1.5 else "running", 0.92)


def test_collect_then_predict_roundtrip(monkeypatch, tmp_path: Path) -> None:
    """Collect packets to disk, reload them, and run predictions through the CLI."""
    import wifi_activity_recognition.hardware as hardware_module

    original_reader = hardware_module.CSIReader
    hardware_module.CSIReader = lambda *_args, **_kwargs: _RoundTripReader()
    try:
        capture_path = tmp_path / "capture.json"
        collect_result = CliRunner().invoke(
            cli,
            [
                "collect",
                "--hardware",
                "esp32",
                "--packets",
                "2",
                "--output",
                str(capture_path),
            ],
        )
        assert collect_result.exit_code == 0, collect_result.output
    finally:
        hardware_module.CSIReader = original_reader

    fake_inference = types.ModuleType("wifi_activity_recognition.inference")
    fake_inference.ActivityRecognizer = _RoundTripRecognizer
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.inference", fake_inference)

    fake_models = types.ModuleType("wifi_activity_recognition.models")
    fake_models.load_model = lambda _path: object()
    monkeypatch.setitem(sys.modules, "wifi_activity_recognition.models", fake_models)

    model_path = tmp_path / "model.pt"
    model_path.write_text("model", encoding="utf-8")
    output_path = tmp_path / "predictions.json"

    predict_result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--hardware",
            "esp32",
            "--model",
            str(model_path),
            "--input",
            str(capture_path),
            "--output",
            str(output_path),
        ],
    )

    assert predict_result.exit_code == 0, predict_result.output
    records = json.loads(output_path.read_text(encoding="utf-8"))
    assert [record["prediction"] for record in records] == ["walking", "running"]
    assert all(record["confidence"] >= 0.9 for record in records)

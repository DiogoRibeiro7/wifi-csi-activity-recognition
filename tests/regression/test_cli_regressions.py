"""Regression tests for previously broken CLI behaviors."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from wifi_activity_recognition.cli import cli
from wifi_activity_recognition.hardware.base import CSIData

pytestmark = [pytest.mark.regression]


class _DummyRecognizer:
    """Small recognizer stub for regression testing."""

    def __init__(self, _model) -> None:
        """Initialize the stub recognizer."""

    def predict(self, _sample: CSIData) -> tuple[str, float]:
        """Return a fixed prediction."""
        return "walking", 0.9


def _packet() -> CSIData:
    """Create one synthetic CSI packet."""
    return CSIData(
        timestamp=0.0,
        amplitude=np.ones((1, 1, 4)),
        phase=np.zeros((1, 1, 4)),
        frequency=2400.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=4,
    )


def test_predict_rejects_array_payloads_regression(
    monkeypatch, tmp_path: Path
) -> None:
    """Predict must reject ndarray payloads instead of pretending they are packets."""
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

    result = CliRunner().invoke(
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


def test_collect_uses_one_stream_session_regression(tmp_path: Path) -> None:
    """Collect must not recreate the stream iterator for every packet."""
    class DummyReader:
        def __init__(self) -> None:
            self.stream_calls = 0
            self._packets = iter([_packet(), _packet()])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def stream(self):
            self.stream_calls += 1
            return self._packets

    reader = DummyReader()

    import wifi_activity_recognition.hardware as hardware_module

    original_reader = hardware_module.CSIReader
    hardware_module.CSIReader = lambda *_args, **_kwargs: reader
    try:
        output_path = tmp_path / "capture.json"
        result = CliRunner().invoke(
            cli,
            [
                "collect",
                "--hardware",
                "esp32",
                "--packets",
                "2",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert reader.stream_calls == 1
    finally:
        hardware_module.CSIReader = original_reader

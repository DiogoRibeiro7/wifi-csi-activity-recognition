# isort: skip_file
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Make the package importable despite repository layout using hyphenated name
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
if "wifi_activity_recognition" not in sys.modules:
    package = types.ModuleType("wifi_activity_recognition")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["wifi_activity_recognition"] = package
    hw = types.ModuleType("wifi_activity_recognition.hardware")
    hw.__path__ = [str(PACKAGE_ROOT / "hardware")]
    sys.modules["wifi_activity_recognition.hardware"] = hw
    package.hardware = hw

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.inference import (  # type: ignore  # noqa: E402
    ActivityRecognizer,
    StreamingPredictor,
    postprocessing as post,
)
from wifi_activity_recognition.models.cnn2d import (  # type: ignore  # noqa: E402
    CNN2DModel,
)


@pytest.fixture()
def csi_packet() -> CSIData:
    amp = np.random.rand(1, 1, 30).astype(np.float32)
    phase = np.random.rand(1, 1, 30).astype(np.float32)
    return CSIData(
        timestamp=0.0,
        amplitude=amp,
        phase=phase,
        frequency=2.4,
        bandwidth=20.0,
        n_tx=1,
        n_rx=1,
        n_subcarriers=30,
    )


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    model = CNN2DModel(num_classes=2)
    path = tmp_path / "model.pth"
    torch.save(model, path)
    return path


def test_activity_recognizer_predict(csi_packet: CSIData, model_file: Path) -> None:
    recognizer = ActivityRecognizer(model_file, class_names=["a", "b"])
    label, conf = recognizer.predict(csi_packet)
    assert label in {"a", "b"}
    assert 0.0 <= conf <= 1.0


def test_streaming_predictor_returns_after_window(
    csi_packet: CSIData, model_file: Path
) -> None:
    recognizer = ActivityRecognizer(model_file, class_names=["a", "b"])
    predictor = StreamingPredictor(recognizer, window_size=3, threshold=0.0)
    assert predictor.update(csi_packet) is None
    assert predictor.update(csi_packet) is None
    result = predictor.update(csi_packet)
    assert result is not None
    label, conf, ts = result
    assert label in {"a", "b"}
    assert 0.0 <= conf <= 1.0
    assert ts == pytest.approx(csi_packet.timestamp)


def test_postprocessing_helpers() -> None:
    probs = [np.array([0.2, 0.8]), np.array([0.6, 0.4])]
    smoothed = post.smooth_probabilities(probs)
    assert smoothed.shape == (2,)
    assert np.allclose(smoothed, np.array([0.4, 0.6]))
    assert post.apply_confidence_threshold(0.7, "x", 0.5) == ("x", 0.7)
    assert post.apply_confidence_threshold(0.4, "x", 0.5) is None

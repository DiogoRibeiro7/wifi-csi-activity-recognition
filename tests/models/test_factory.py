"""Tests for model factory helpers."""

from wifi_activity_recognition.models.factory import list_available_models


def test_list_available_models_returns_registered_metadata():
    """The factory should expose stable metadata for registered models."""
    models = list_available_models()

    assert "cnn2d" in models
    assert models["cnn2d"]["class_name"] == "CNN2DModel"
    assert models["cnn2d"]["description"]
    assert "transformer" in models

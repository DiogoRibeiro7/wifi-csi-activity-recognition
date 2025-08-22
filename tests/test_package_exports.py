"""Regression tests for package-level exports."""

import importlib


def test_hardware_submodule_available() -> None:
    """Importing the hardware module registers it on the package."""
    pkg = importlib.import_module("wifi_activity_recognition")
    importlib.import_module("wifi_activity_recognition.hardware")
    assert hasattr(pkg, "hardware"), "hardware submodule not registered"

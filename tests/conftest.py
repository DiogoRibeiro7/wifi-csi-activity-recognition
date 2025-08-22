"""Common test configuration for wifi-csi-activity-recognition.

This ensures the real package ``wifi-activity-recognition`` (with a hyphen
in its name on disk) can be imported as ``wifi_activity_recognition`` during
tests by loading its ``__init__`` module via a custom import spec. This
executes the package's initialization code so attributes like ``hardware``
are registered as they would be in a normal installation.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "wifi-activity-recognition"

# Skip heavy dependency checks during test collection
os.environ.setdefault("WIFI_AR_SKIP_DEPS", "1")

# Load the actual package module so its __init__ executes
spec = importlib.util.spec_from_file_location(
    "wifi_activity_recognition", PACKAGE_ROOT / "__init__.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules.setdefault("wifi_activity_recognition", module)
assert spec.loader is not None  # for type checkers
spec.loader.exec_module(module)

# Preload hardware submodule so attribute is always available
importlib.import_module("wifi_activity_recognition.hardware")

# Ensure tests that manipulate sys.modules don't break the alias
MODULE = module


@pytest.fixture(autouse=True)
def _restore_package() -> None:
    """Restore the wifi_activity_recognition package for each test."""
    sys.modules["wifi_activity_recognition"] = MODULE

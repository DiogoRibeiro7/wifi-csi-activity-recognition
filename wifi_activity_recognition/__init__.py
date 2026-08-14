"""Top-level package for WiFi activity recognition."""

from __future__ import annotations

import importlib
import os as _os
from typing import Any

from .version import __version__

__all__ = [
    "__version__",
    "CSIReader",
    "ActivityRecognizer",
    "StreamingPredictor",
    "Dataset",
    "Trainer",
    "list_supported_hardware",
    "load_public_dataset",
    "load_config",
    "get_default_config",
    "models",
    "preprocessing",
    "features",
    "hardware",
    "utils",
    "research",
    "multimodal",
]

__author__ = "Diogo Ribeiro"
__email__ = "dfr@esmad.ipp.pt"
__license__ = "MIT"
__description__ = "WiFi CSI-based human activity recognition using computer vision"
__url__ = "https://github.com/diogoribeiro7/wifi-csi-activity-recognition"

SUPPORTED_ACTIVITIES = [
    "walking",
    "running",
    "sitting",
    "standing",
    "lying_down",
    "waving",
    "pointing",
    "fall_detection",
    "no_activity",
]

# Canonical names of the drivers actually registered at import time. Aliases
# such as "intel5300" and "atheros" also resolve; see the hardware registry.
# Kept in step with the registry by
# tests/hardware/test_supported_hardware_claims.py.
SUPPORTED_HARDWARE = [
    "intel_5300",
    "esp32",
    "atheros_ar9300",
    "qualcomm",
]

# Platforms that are documented as future work and have no driver module. These
# are deliberately NOT reported as supported: advertising them made the public
# API claim capability the package does not have.
PLANNED_HARDWARE = [
    "broadcom",
    "mediatek",
]

_SUBMODULE_ALIASES = {
    "features": ".features",
    "hardware": ".hardware",
    "models": ".models",
    "multimodal": ".multimodal",
    "preprocessing": ".preprocessing",
    "research": ".research",
    "utils": ".utils",
}

_LAZY_EXPORTS = {
    "Dataset": (".datasets", "Dataset"),
    "load_public_dataset": (".datasets", "load_public_dataset"),
    "CSIReader": (".hardware", "CSIReader"),
    "list_supported_hardware": (".hardware", "list_supported_hardware"),
    "ActivityRecognizer": (".inference", "ActivityRecognizer"),
    "StreamingPredictor": (".inference", "StreamingPredictor"),
    "Trainer": (".training", "Trainer"),
    "load_config": (".utils.config", "load_config"),
    "get_default_config": (".utils.config", "get_default_config"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose heavy submodules and convenience exports."""
    if name in _SUBMODULE_ALIASES:
        module = importlib.import_module(_SUBMODULE_ALIASES[name], __name__)
        globals()[name] = module
        return module

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return module attributes exposed by the lazy top-level API."""
    return sorted(set(globals()) | set(__all__))


def get_package_info() -> dict[str, Any]:
    """Get comprehensive package information."""
    return {
        "version": __version__,
        "supported_hardware": SUPPORTED_HARDWARE,
        "planned_hardware": PLANNED_HARDWARE,
        "supported_activities": SUPPORTED_ACTIVITIES,
        "author": __author__,
        "license": __license__,
        "url": __url__,
    }


def check_dependencies() -> None:
    """Check whether the declared core dependencies can be imported."""
    required_deps = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "cv2",
        "matplotlib",
        "yaml",
        "h5py",
        "networkx",
        "pywt",
    ]

    missing_required = []

    for dep in required_deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_required.append(dep)

    if missing_required:
        raise ImportError(
            f"Missing required dependencies: {missing_required}. "
            "Install with: pip install wifi-activity-recognition"
        )


if _os.environ.get("WIFI_AR_SKIP_DEPS") != "1":
    check_dependencies()

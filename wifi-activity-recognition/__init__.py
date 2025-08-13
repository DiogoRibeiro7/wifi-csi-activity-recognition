"""
WiFi Activity Recognition Package

A comprehensive Python package for human activity recognition using WiFi Channel State
Information (CSI) and computer vision techniques.

This package provides:
- Hardware abstraction for various WiFi platforms (Intel 5300, ESP32, etc.)
- Standardized CSI data processing pipeline
- Computer vision models optimized for CSI data
- Real-time activity recognition capabilities
- Pre-trained models for common activities

Basic Usage:
    >>> from wifi_activity_recognition import CSIReader, ActivityRecognizer
    >>> from wifi_activity_recognition.models import load_pretrained_model
    >>>
    >>> # Initialize hardware reader
    >>> reader = CSIReader('esp32', config={'sampling_rate': 100})
    >>>
    >>> # Load pre-trained model
    >>> model = load_pretrained_model('general_activities_v1')
    >>> recognizer = ActivityRecognizer(model)
    >>>
    >>> # Real-time recognition
    >>> for csi_data in reader.stream():
    ...     activity, confidence = recognizer.predict(csi_data)
    ...     print(f"Activity: {activity} ({confidence:.2f})")

For more examples and detailed documentation, visit:
https://wifi-activity-recognition.readthedocs.io/
"""

from .version import __version__

# Core components
from .hardware import CSIReader, list_supported_hardware
from .inference import ActivityRecognizer, StreamingPredictor
from .datasets import Dataset, load_public_dataset
from .training import Trainer

# Models and utilities
from . import models
from . import preprocessing
from . import features
from . import utils

# Configuration
from .utils.config import load_config, get_default_config

__all__ = [
    # Version
    "__version__",
    # Core classes
    "CSIReader",
    "ActivityRecognizer",
    "StreamingPredictor",
    "Dataset",
    "Trainer",
    # Utility functions
    "list_supported_hardware",
    "load_public_dataset",
    "load_config",
    "get_default_config",
    # Submodules
    "models",
    "preprocessing",
    "features",
    "utils",
]

# Package metadata
__author__ = "Your Name"
__email__ = "your.email@domain.com"
__license__ = "MIT"
__description__ = "WiFi CSI-based human activity recognition using computer vision"
__url__ = "https://github.com/yourusername/wifi-activity-recognition"

# Supported activities (can be extended)
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

# Supported hardware platforms
SUPPORTED_HARDWARE = [
    "intel_5300",
    "esp32",
    "atheros_ar9300",
    "qualcomm",
    "broadcom",
    "mediatek",
]


def get_package_info():
    """Get comprehensive package information."""
    return {
        "version": __version__,
        "supported_hardware": SUPPORTED_HARDWARE,
        "supported_activities": SUPPORTED_ACTIVITIES,
        "author": __author__,
        "license": __license__,
        "url": __url__,
    }


def check_dependencies():
    """Check if required dependencies are installed."""
    import importlib

    required_deps = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "cv2",
        "matplotlib",
        "yaml",
        "h5py",
    ]

    optional_deps = {
        "torch": "PyTorch backend for neural networks",
        "tensorflow": "TensorFlow backend for neural networks",
    }

    missing_required = []
    missing_optional = []

    for dep in required_deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_required.append(dep)

    for dep, description in optional_deps.items():
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_optional.append((dep, description))

    if missing_required:
        raise ImportError(
            f"Missing required dependencies: {missing_required}. "
            f"Install with: pip install wifi-activity-recognition"
        )

    if missing_optional:
        import warnings

        for dep, desc in missing_optional:
            warnings.warn(
                f"Optional dependency '{dep}' not found. {desc}. "
                f"Install with: pip install wifi-activity-recognition[{dep}]"
            )


# Run dependency check on import
check_dependencies()

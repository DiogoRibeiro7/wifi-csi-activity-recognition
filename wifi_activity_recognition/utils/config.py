"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "hardware": {"type": "esp32"},
    "model": {"type": "cnn2d"},
    "training": {"batch_size": 32, "learning_rate": 1e-3},
}


def get_default_config() -> dict[str, Any]:
    """Return a copy of the package's default configuration."""
    return DEFAULT_CONFIG.copy()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with environment variable expansion.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary. Returns an empty dict when the file is
        empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = os.path.expandvars(f.read())
    return yaml.safe_load(text) or {}


def validate_config(config: Mapping[str, Any], schema_path: str | Path) -> None:
    """Validate a configuration dictionary against a simple YAML schema.

    The schema should mirror the structure of the configuration and list
    required keys. Nested dictionaries are validated recursively.
    """
    schema = load_config(schema_path)
    _validate_dict(config, schema, path="")


def merge_configs(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return the result."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def _validate_dict(
    config: Mapping[str, Any], schema: Mapping[str, Any], path: str
) -> None:
    """Recursively validate dictionary keys."""
    for key, subschema in schema.items():
        current_path = f"{path}{key}"
        if key not in config:
            raise ValueError(f"Missing required key: {current_path}")
        if isinstance(subschema, Mapping):
            if not isinstance(config[key], Mapping):
                raise ValueError(f"Key '{current_path}' should contain a mapping")
            _validate_dict(config[key], subschema, path=f"{current_path}.")

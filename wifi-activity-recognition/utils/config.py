"""YAML configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path: str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_config(config: Mapping[str, Any], schema_path: str | Path) -> None:
    """Validate a configuration dictionary against a simple YAML schema.

    The schema should mirror the structure of the configuration and list
    required keys. Nested dictionaries are validated recursively.

    Parameters
    ----------
    config: Mapping[str, Any]
        Configuration data to validate.
    schema_path: str or Path
        Path to the YAML schema describing required keys.

    Raises
    ------
    ValueError
        If a required key is missing from the configuration.
    """
    schema = load_config(schema_path)
    _validate_dict(config, schema, path="")


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

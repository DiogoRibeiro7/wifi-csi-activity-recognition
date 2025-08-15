import sys
import types
from pathlib import Path

import pytest
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.utils.config import (  # type: ignore  # noqa: E402
    get_default_config,
    load_config,
    validate_config,
)


def test_load_and_validate_config(tmp_path):
    config = {"db": {"host": "localhost", "port": 8080}}
    schema = {"db": {"host": None, "port": None}}
    config_path = tmp_path / "config.yml"
    schema_path = tmp_path / "schema.yml"
    config_path.write_text(yaml.safe_dump(config))
    schema_path.write_text(yaml.safe_dump(schema))

    loaded = load_config(config_path)
    assert loaded == config
    validate_config(loaded, schema_path)

    incomplete = {"db": {"host": "localhost"}}
    with pytest.raises(ValueError):
        validate_config(incomplete, schema_path)


def test_get_default_config():
    cfg = get_default_config()
    assert "hardware" in cfg and "model" in cfg

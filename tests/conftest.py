"""Common pytest configuration for repository-local test runs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Skip heavy dependency checks during test collection.
os.environ.setdefault("WIFI_AR_SKIP_DEPS", "1")

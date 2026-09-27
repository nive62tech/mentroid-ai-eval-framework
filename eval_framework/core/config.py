"""
Lightweight YAML/JSON config loader for evaluation runs.

Keeping model/dataset/threshold settings in a config file (rather than
hardcoded in scripts) is what makes runs reproducible and comparable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_config(path: str) -> Dict[str, Any]:
    """Load a .yaml/.yml or .json config file into a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            if not _HAS_YAML:
                raise ImportError(
                    "PyYAML is required to load YAML configs. "
                    "Install with: pip install pyyaml --break-system-packages"
                )
            return yaml.safe_load(f) or {}
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {ext}")

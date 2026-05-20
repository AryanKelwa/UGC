"""
src/utils/config_loader.py
==========================
Configuration loader that parses YAML files.
Provides dotted-attribute access to nested settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
import yaml

log = logging.getLogger("pipeline.utils.config")

class DotDict(dict):
    """A dictionary subclass that allows dot notation to access keys as attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct: dict | None = None) -> None:
        super().__init__()
        if dct is not None:
            for key, value in dct.items():
                if isinstance(value, dict):
                    self[key] = DotDict(value)
                elif isinstance(value, list):
                    self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]
                else:
                    self[key] = value


def load_yaml_config(config_path: Path | str) -> DotDict:
    """Load a YAML configuration file and return it as a DotDict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_dct = yaml.safe_load(f) or {}
            return DotDict(raw_dct)
        except yaml.YAMLError as exc:
            log.error("Failed to parse YAML configuration: %s", exc)
            raise

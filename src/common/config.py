from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_cfg(path: str | Path) -> dict:
    """Load YAML config. Compatibility function expected by newer scripts."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
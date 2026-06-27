"""Config loader. Returns the parsed YAML as a dict — no settings class until a
value actually needs validation (ponytail: a dict is enough here)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("POLYCRAWLER_CONFIG", DEFAULT))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

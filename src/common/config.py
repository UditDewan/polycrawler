"""Config loader. Returns the parsed YAML as a dict — no settings class until a
value actually needs validation (ponytail: a dict is enough here)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader (no dependency). Sets only vars not already in os.environ."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None) -> dict:
    load_env()  # so NVIDIA_API_KEY etc. from .env are visible to clients
    path = Path(path or os.environ.get("POLYCRAWLER_CONFIG", DEFAULT))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

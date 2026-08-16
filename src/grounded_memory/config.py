"""Lightweight YAML config loader with environment-variable interpolation.

Kept dependency-light on purpose (PyYAML only). Supports the OmegaConf-style
token ``${oc.env:VAR,default}`` so the same config files remain compatible if
the project later adopts OmegaConf/Hydra.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")

# Repository root = three levels up from this file (src/grounded_memory/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _interpolate_env(value: str) -> str:
    """Resolve ``${oc.env:VAR,default}`` tokens inside a string."""

    def repl(match: re.Match[str]) -> str:
        var = match.group(1).strip()
        default = match.group(2)
        return os.environ.get(var, default if default is not None else "")

    return _ENV_PATTERN.sub(repl, value)


def _resolve(obj: Any) -> Any:
    """Recursively resolve env interpolation across a nested structure."""
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v) for v in obj]
    if isinstance(obj, str):
        return _interpolate_env(obj)
    return obj


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by bare name (looked up in ``configs/``) or explicit path.

    Examples
    --------
    >>> cfg = load_config("evaluation")          # loads configs/evaluation.yaml
    >>> cfg = load_config("configs/models.yaml")  # explicit path
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{path.name}.yaml"
    if not path.is_absolute():
        # Try as-given first, then relative to the repo root.
        candidate = path if path.exists() else (REPO_ROOT / path)
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _resolve(raw)


__all__ = ["load_config", "REPO_ROOT", "CONFIGS_DIR"]

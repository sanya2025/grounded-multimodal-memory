"""Experiment tracking: provenance-complete prediction records + JSONL writer.

Every saved prediction records model, prompt, generation settings, dataset
version, git commit, latency, tokens, and metrics so results are reproducible and
raw predictions stay separate from derived metrics. Never overwrite prior outputs
by default (the writer refuses to clobber unless ``overwrite=True``).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(default: str = "unknown") -> str:
    """Best-effort current git commit SHA (short)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or default
    except Exception:  # noqa: BLE001 - git may be absent; tracking must not crash
        return default


def hardware_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:  # optional torch/GPU probe
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


@dataclass
class PredictionRecord:
    """One row of raw prediction output with full provenance."""

    experiment_id: str
    run_id: str
    sample_id: str
    model_id: str
    prompt_id: str
    prompt_text: str
    prediction: str
    # provenance / settings
    model_revision: str = ""
    model_dtype: str = ""
    quantization: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 512
    dataset: str = ""
    dataset_version: str = ""
    manifest_version: str = ""
    parsed_prediction: dict[str, Any] | None = None
    parse_ok: bool | None = None
    ground_truth: Any = None
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Backend-specific extras from GenerationOutput.metadata (e.g. Ollama's
    # thinking_fallback / done_reason) -- preserved so nothing is silently
    # dropped, per this project's "never fabricate, keep raw provenance" rule.
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    git_commit: str = field(default_factory=git_commit)
    hardware: dict[str, str] = field(default_factory=hardware_info)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def write_jsonl(
    records: list[PredictionRecord], path: str | Path, overwrite: bool = False
) -> Path:
    """Append prediction records to a JSONL file.

    Refuses to overwrite an existing file unless ``overwrite=True`` (protects
    frozen raw predictions). Creates parent dirs as needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"{p} exists. Refusing to overwrite frozen predictions "
            f"(pass overwrite=True to force, or write to a new run_id path)."
        )
    mode = "w" if overwrite else "a"
    with p.open(mode, encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")
    return p


def make_run_id(prefix: str = "run") -> str:
    """Deterministic-ish run id from env or timestamp (no global RNG)."""
    stamp = os.environ.get("GMM_RUN_ID")
    if stamp:
        return stamp
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


__all__ = [
    "PredictionRecord",
    "write_jsonl",
    "git_commit",
    "hardware_info",
    "make_run_id",
]

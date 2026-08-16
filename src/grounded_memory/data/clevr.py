"""CLEVR loader for the ~30-example controlled diagnostic (E1.4).

Synthetic scenes with ground-truth object locations, attributes, relationships,
questions, answers, and functional programs. Appendix/diagnostic only — never the
main blog benchmark. https://cs.stanford.edu/people/jcjohns/clevr/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_memory.config import load_config


@dataclass
class ClevrExample:
    image_filename: str
    question: str
    answer: str
    program: list[dict[str, Any]]
    scene: dict[str, Any] | None = None


def load_diagnostic(n: int | None = None, datasets_cfg: dict | None = None) -> list[ClevrExample]:
    """Load the first ``n`` CLEVR questions (default from config)."""
    cfg = datasets_cfg or load_config("datasets")
    clevr = cfg["clevr"]
    n = n or clevr.get("diagnostic_size", 30)
    qpath = Path(clevr["root"]) / clevr["questions"]
    if not qpath.exists():
        raise FileNotFoundError(
            f"CLEVR questions not found: {qpath}. Download CLEVR and set "
            f"configs/datasets.yaml -> clevr.root."
        )
    questions = json.loads(qpath.read_text()).get("questions", [])[:n]
    return [
        ClevrExample(
            image_filename=q.get("image_filename", ""),
            question=q.get("question", ""),
            answer=str(q.get("answer", "")),
            program=q.get("program", []),
        )
        for q in questions
    ]


__all__ = ["ClevrExample", "load_diagnostic"]

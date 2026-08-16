"""Parsing structured model output into a validated ``SceneRepresentation``.

A model returns free text. Condition C asks for JSON. Real models emit JSON with
markdown fences, trailing prose, or minor schema drift. We parse *robustly* and,
crucially, we **never throw away a prediction** just because parsing failed:
callers get a ``ParseResult`` that always retains the raw text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from grounded_memory.scene.schema import SceneRepresentation

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    """Outcome of parsing one structured prediction.

    ``parse_ok`` distinguishes *parser* failure from *model* failure so the two
    can be accounted for separately (Execution philosophy: separate error types).
    """

    raw_output: str
    scene: SceneRepresentation | None
    parse_ok: bool
    error: str | None = None


def _extract_json_blob(text: str) -> str | None:
    """Best-effort extraction of the first JSON object from model text."""
    if not text:
        return None
    # 1) Fenced ```json ... ``` block.
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2) First balanced {...} span.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_scene_output(raw_output: str) -> ParseResult:
    """Parse raw model text into a ``SceneRepresentation`` without ever raising.

    Returns a ``ParseResult`` whose ``scene`` is ``None`` on failure while always
    preserving ``raw_output``.
    """
    blob = _extract_json_blob(raw_output)
    if blob is None:
        return ParseResult(raw_output, None, False, "no JSON object found")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return ParseResult(raw_output, None, False, f"json decode error: {exc}")
    try:
        scene = SceneRepresentation.model_validate(data)
    except ValidationError as exc:
        return ParseResult(raw_output, None, False, f"schema validation error: {exc}")
    return ParseResult(raw_output, scene, True, None)


__all__ = ["ParseResult", "parse_scene_output"]

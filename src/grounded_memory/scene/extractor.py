"""Run a VLM under a chosen prompt condition to extract a scene representation.

This is the shared entry point for Experiment 1 and for the /observe API path.
It preserves raw output and parser status so nothing is silently discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from grounded_memory.models.base import GenerationOutput, VisionLanguageModel
from grounded_memory.prompts import get_prompt
from grounded_memory.scene.claims import ParseResult, parse_scene_output
from grounded_memory.scene.schema import SceneRepresentation


@dataclass
class SceneExtraction:
    """Everything produced by one (image, model, condition) extraction."""

    condition: str
    prompt_id: str
    prompt_text: str
    raw_output: str
    scene: SceneRepresentation | None
    parse_ok: bool
    parse_error: str | None
    generation: GenerationOutput


def extract_scene(
    model: VisionLanguageModel,
    image: Any,
    condition: str = "structured",
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> SceneExtraction:
    """Extract a scene under one prompt condition.

    For the ``structured`` condition the output is parsed into a
    ``SceneRepresentation``; for ``standard``/``grounded`` the raw text is kept
    and ``scene`` is left ``None`` (natural-language conditions are scored by
    their text, not by JSON).
    """
    prompt = get_prompt(condition)
    gen = model.generate(image, prompt.text, temperature=temperature, max_new_tokens=max_new_tokens)

    if condition == "structured":
        parsed: ParseResult = parse_scene_output(gen.text)
        scene, ok, err = parsed.scene, parsed.parse_ok, parsed.error
    else:
        scene, ok, err = None, True, None

    return SceneExtraction(
        condition=condition,
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.text,
        raw_output=gen.text,
        scene=scene,
        parse_ok=ok,
        parse_error=err,
        generation=gen,
    )


__all__ = ["SceneExtraction", "extract_scene"]

"""Experiment-1 pipeline: run one image under the three prompt conditions.

Produces provenance-complete ``PredictionRecord`` objects (raw output preserved,
parser status recorded). Metrics are computed downstream from the frozen raw
predictions, not here (separate raw predictions from derived metrics).
"""

from __future__ import annotations

from typing import Any

from grounded_memory.models.base import VisionLanguageModel
from grounded_memory.prompts import CONDITIONS
from grounded_memory.scene.extractor import extract_scene
from grounded_memory.tracking import PredictionRecord


def run_conditions_for_image(
    model: VisionLanguageModel,
    image: Any,
    sample_id: str,
    experiment_id: str,
    run_id: str,
    conditions: list[str] | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    dataset: str = "gqa",
    dataset_version: str = "v1",
    manifest_version: str = "v1",
) -> list[PredictionRecord]:
    """Run all requested prompt conditions for one image; return raw records."""
    conditions = conditions or list(CONDITIONS.keys())
    records: list[PredictionRecord] = []
    for condition in conditions:
        ext = extract_scene(
            model, image, condition=condition,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        gen = ext.generation
        records.append(
            PredictionRecord(
                experiment_id=experiment_id,
                run_id=run_id,
                sample_id=sample_id,
                model_id=model.model_id,
                model_revision=model.revision,
                prompt_id=ext.prompt_id,
                prompt_text=ext.prompt_text,
                prediction=ext.raw_output,
                parsed_prediction=ext.scene.model_dump() if ext.scene else None,
                parse_ok=ext.parse_ok,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                dataset=dataset,
                dataset_version=dataset_version,
                manifest_version=manifest_version,
                latency_s=gen.latency_s,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                metrics={"condition": condition, "parse_error": ext.parse_error},
            )
        )
    return records


__all__ = ["run_conditions_for_image"]

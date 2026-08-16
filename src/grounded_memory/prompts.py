"""Prompt definitions for Experiment 1 (three grounding conditions).

Every prompt is versioned by a stable ``prompt_id`` so predictions can record
exactly which prompt produced them (Reproducibility rule #3: save every prompt
verbatim). Do not edit a prompt's text in place after results are frozen; add a
new versioned entry instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    condition: str          # "standard" | "grounded" | "structured"
    text: str


# --- Condition A: Standard ---
STANDARD = Prompt(
    prompt_id="e1_standard_v1",
    condition="standard",
    text="Describe what is happening in this image.",
)

# --- Condition B: Grounded natural language ---
GROUNDED = Prompt(
    prompt_id="e1_grounded_v1",
    condition="grounded",
    text=(
        "Analyze this image carefully.\n\n"
        "Identify:\n"
        "1. visible entities,\n"
        "2. important attributes,\n"
        "3. relationships,\n"
        "4. spatial relationships,\n"
        "5. human actions,\n"
        "6. evidence supporting each important claim.\n\n"
        "Clearly distinguish directly observable evidence from inference.\n\n"
        "If a conclusion cannot be confidently determined from the image, "
        "state that the available visual evidence is insufficient."
    ),
)

# --- Condition C: Structured grounded output (validated against SceneRepresentation) ---
STRUCTURED = Prompt(
    prompt_id="e1_structured_v1",
    condition="structured",
    text=(
        "Analyze this image and return ONLY a single JSON object (no prose, no "
        "markdown fences) conforming to this schema:\n\n"
        "{\n"
        '  "scene_summary": string,\n'
        '  "entities": [{"id": string, "label": string, "attributes": [string], '
        '"confidence": number}],\n'
        '  "relationships": [{"subject": string, "relation": string, '
        '"object": string, "confidence": number}],\n'
        '  "spatial_relations": [{"subject": string, "relation": string, '
        '"object": string, "confidence": number}],\n'
        '  "claims": [{"claim": string, "evidence": string, '
        '"evidence_type": "direct_or_supported"|"inferred"|"unsupported"'
        '|"insufficient_evidence", "confidence": number}],\n'
        '  "uncertainty": [{"about": string, "reason": string}]\n'
        "}\n\n"
        "Rules:\n"
        "- Use entity ids like 'person_1', 'object_1'.\n"
        "- For every factual claim, set evidence_type to distinguish what is "
        "directly observable from what is inferred.\n"
        "- If something cannot be determined from the image, add it to "
        "'uncertainty' and do NOT assert it as a claim.\n"
        "- Report confidence as model-reported only (it is NOT calibrated)."
    ),
)

CONDITIONS: dict[str, Prompt] = {
    STANDARD.condition: STANDARD,
    GROUNDED.condition: GROUNDED,
    STRUCTURED.condition: STRUCTURED,
}


def get_prompt(condition: str) -> Prompt:
    """Return the prompt for a condition ('standard' | 'grounded' | 'structured')."""
    try:
        return CONDITIONS[condition]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt condition {condition!r}; expected one of {list(CONDITIONS)}"
        ) from exc


# --- Memory / temporal QA prompt (Experiment 2) ---
def build_memory_qa_prompt(question: str, retrieved_context: str) -> str:
    """Assemble a grounded temporal-QA prompt from retrieved memory events."""
    return (
        "You answer questions about a sequence of past visual observations.\n"
        "Use ONLY the retrieved observations below. If they do not contain the "
        "answer, reply exactly: 'The available visual observations are insufficient.'\n\n"
        f"Retrieved observations:\n{retrieved_context}\n\n"
        f"Question: {question}\n"
        "Answer with the fact and cite the relevant observation id(s)."
    )


__all__ = [
    "Prompt",
    "STANDARD",
    "GROUNDED",
    "STRUCTURED",
    "CONDITIONS",
    "get_prompt",
    "build_memory_qa_prompt",
]

"""Tri-state evidence verification of a claim against a scene-graph reference.

CRITICAL: GQA/Visual-Genome scene graphs can be incomplete. A claim that is not
present in the graph is ``NOT_VERIFIABLE`` — NOT automatically a hallucination.
Only a claim that contradicts an annotated fact is ``CONTRADICTED``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from grounded_memory.evaluation.normalize import normalize_object, normalize_relation


class EvidenceLabel(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_VERIFIABLE = "not_verifiable"


def verify_object_claim(obj_label: str, scene_graph: dict[str, Any]) -> EvidenceLabel:
    """Is an object-presence claim supported by the scene graph?

    Supported if the object is annotated. If not annotated we return
    NOT_VERIFIABLE (absence in an incomplete graph is not proof of absence).
    """
    present = {
        normalize_object(o.get("name", ""))
        for o in scene_graph.get("objects", {}).values()
        if o.get("name")
    }
    return (
        EvidenceLabel.SUPPORTED
        if normalize_object(obj_label) in present
        else EvidenceLabel.NOT_VERIFIABLE
    )


def verify_relation_claim(
    subject: str, relation: str, obj: str, scene_graph: dict[str, Any]
) -> EvidenceLabel:
    """Verify a (subject, relation, object) claim tri-state.

    - SUPPORTED: the exact normalized triple is annotated.
    - CONTRADICTED: subject & object are annotated with this relation but to a
      different partner (an annotated conflict), handled by callers that build
      counterfactuals; here we conservatively return NOT_VERIFIABLE unless the
      triple is present.
    """
    objects = scene_graph.get("objects", {})
    by_name = {normalize_object(o.get("name", "")): oid for oid, o in objects.items()}
    subj_norm, obj_norm, rel_norm = (
        normalize_object(subject),
        normalize_object(obj),
        normalize_relation(relation),
    )
    subj_id = by_name.get(subj_norm)
    if subj_id is None:
        return EvidenceLabel.NOT_VERIFIABLE
    for rel in objects.get(subj_id, {}).get("relations", []):
        tgt = objects.get(rel.get("object", ""), {})
        if (
            normalize_relation(rel.get("name", "")) == rel_norm
            and normalize_object(tgt.get("name", "")) == obj_norm
        ):
            return EvidenceLabel.SUPPORTED
    return EvidenceLabel.NOT_VERIFIABLE


__all__ = ["EvidenceLabel", "verify_object_claim", "verify_relation_claim"]

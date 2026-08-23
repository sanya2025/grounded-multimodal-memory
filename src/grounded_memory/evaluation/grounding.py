"""Tri-state evidence verification of a claim against a scene-graph reference.

CRITICAL: GQA/Visual-Genome scene graphs can be incomplete. A claim that is not
present in the graph is ``NOT_VERIFIABLE`` — NOT automatically a hallucination.
Only a claim that contradicts an annotated fact is ``CONTRADICTED``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from grounded_memory.evaluation.normalize import (
    attribute_category,
    normalize_object,
    normalize_relation,
    normalize_simple,
    spatial_opposite,
)


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
    - CONTRADICTED: the same (subject, object) pair is annotated with the
      OPPOSITE spatial relation (configs/evaluation.yaml
      'spatial_relations.opposites', e.g. predicted "left_of" when GQA
      annotates "right_of" for the same pair -- a real, detectable conflict.
      Only defined for spatial relations; non-spatial relations (holding,
      wearing, ...) have no configured opposite, so they stay NOT_VERIFIABLE
      when unmatched, same as before.
    - NOT_VERIFIABLE: neither the claim nor its opposite is annotated (an
      incomplete graph is not proof the claim is false).
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

    annotated_rels_to_obj = {
        normalize_relation(rel.get("name", ""))
        for rel in objects.get(subj_id, {}).get("relations", [])
        if normalize_object(objects.get(rel.get("object", ""), {}).get("name", "")) == obj_norm
    }
    if rel_norm in annotated_rels_to_obj:
        return EvidenceLabel.SUPPORTED
    opposite = spatial_opposite(rel_norm)
    if opposite is not None and opposite in annotated_rels_to_obj:
        return EvidenceLabel.CONTRADICTED
    return EvidenceLabel.NOT_VERIFIABLE


def verify_attribute_claim(
    obj_label: str, attribute: str, scene_graph: dict[str, Any]
) -> EvidenceLabel:
    """Verify an (object, attribute) claim tri-state.

    - SUPPORTED: the object is annotated with this exact attribute.
    - CONTRADICTED: the object is annotated with a DIFFERENT attribute from
      the same mutually-exclusive category (configs/evaluation.yaml
      'attribute_categories', e.g. predicted "red" when GQA annotates
      "blue" -- color is mutually exclusive for a given object).
    - NOT_VERIFIABLE: object not annotated, attribute has no configured
      category, or no conflicting same-category attribute is present.
    """
    objects = scene_graph.get("objects", {})
    obj_norm = normalize_object(obj_label)
    matching = [o for o in objects.values() if normalize_object(o.get("name", "")) == obj_norm]
    if not matching:
        return EvidenceLabel.NOT_VERIFIABLE

    annotated = {normalize_simple(a) for o in matching for a in (o.get("attributes") or [])}
    attr_norm = normalize_simple(attribute)
    if attr_norm in annotated:
        return EvidenceLabel.SUPPORTED

    category = attribute_category(attr_norm)
    if category is not None:
        for a in annotated:
            if a != attr_norm and attribute_category(a) == category:
                return EvidenceLabel.CONTRADICTED
    return EvidenceLabel.NOT_VERIFIABLE


__all__ = [
    "EvidenceLabel",
    "verify_object_claim",
    "verify_relation_claim",
    "verify_attribute_claim",
]

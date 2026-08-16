"""Generate diagnostic probes directly from a GQA/Visual-Genome scene graph.

We do NOT rely only on the original dataset questions. Given a scene graph we
synthesize positive probes (object/attribute/relation/spatial/compound) and
counterfactual/negative probes (wrong attribute, reversed spatial relation,
absent object). Each probe preserves the source scene-graph facts used to
generate it, so evaluation can distinguish supported / contradicted /
not-verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from grounded_memory.config import load_config


class ProbeType(str, Enum):
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    RELATIONSHIP = "relationship"
    SPATIAL = "spatial"
    COMPOUND_CLAIM = "compound_claim"


@dataclass
class Probe:
    """A single generated diagnostic question."""

    probe_type: ProbeType
    question: str
    expected_answer: str            # "yes"/"no" or a literal (e.g. a color)
    is_counterfactual: bool
    requires_abstention: bool = False
    source_facts: dict[str, Any] = field(default_factory=dict)


def _spatial_opposite(relation: str, cfg: dict | None = None) -> str | None:
    cfg = cfg or load_config("evaluation")
    return cfg.get("spatial_relations", {}).get("opposites", {}).get(relation)


def generate_probes(scene_graph: dict[str, Any], eval_cfg: dict | None = None) -> list[Probe]:
    """Generate positive and counterfactual probes from a scene-graph dict.

    Expected (GQA-style) ``scene_graph`` shape::

        {
          "objects": {
             "obj1": {"name": "umbrella", "attributes": ["black"],
                      "relations": [{"name": "to the left of", "object": "obj2"}]},
             "obj2": {"name": "bicycle", "attributes": [], "relations": []},
             "person1": {"name": "person", "attributes": [],
                         "relations": [{"name": "holding", "object": "obj1"}]}
          }
        }

    The function is defensive about missing keys so it degrades gracefully on
    partial annotations.
    """
    eval_cfg = eval_cfg or load_config("evaluation")
    objects: dict[str, Any] = scene_graph.get("objects", {})
    probes: list[Probe] = []
    present_names = {o.get("name", "").lower() for o in objects.values() if o.get("name")}

    for oid, obj in objects.items():
        name = obj.get("name", "").lower()
        if not name:
            continue

        # --- Object presence (positive) ---
        probes.append(
            Probe(
                ProbeType.OBJECT,
                f"Is there a {name} in the image?",
                "yes",
                is_counterfactual=False,
                source_facts={"object_id": oid, "name": name},
            )
        )

        # --- Attribute (positive + counterfactual) ---
        attrs = [a.lower() for a in obj.get("attributes", [])]
        if attrs:
            attr = attrs[0]
            probes.append(
                Probe(
                    ProbeType.ATTRIBUTE,
                    f"What {_attr_dimension(attr)} is the {name}?",
                    attr,
                    is_counterfactual=False,
                    source_facts={"object_id": oid, "name": name, "attribute": attr},
                )
            )
            probes.append(
                Probe(
                    ProbeType.ATTRIBUTE,
                    f"Is the {name} {_counterfactual_attr(attr)}?",
                    "no",
                    is_counterfactual=True,
                    source_facts={"object_id": oid, "name": name, "true_attribute": attr},
                )
            )

        # --- Relations & spatial ---
        for rel in obj.get("relations", []):
            rname = rel.get("name", "").lower()
            tgt = objects.get(rel.get("object", ""), {})
            tgt_name = tgt.get("name", "").lower()
            if not rname or not tgt_name:
                continue
            canonical = _canonical_spatial(rname, eval_cfg)
            if canonical:
                # Spatial positive.
                probes.append(
                    Probe(
                        ProbeType.SPATIAL,
                        f"Is the {name} {rname} the {tgt_name}?",
                        "yes",
                        is_counterfactual=False,
                        source_facts={"subject": name, "relation": canonical, "object": tgt_name},
                    )
                )
                # Spatial counterfactual: reverse the relation.
                opp = _spatial_opposite(canonical, eval_cfg)
                if opp:
                    probes.append(
                        Probe(
                            ProbeType.SPATIAL,
                            f"Is the {name} {_surface_spatial(opp)} the {tgt_name}?",
                            "no",
                            is_counterfactual=True,
                            source_facts={"subject": name, "true_relation": canonical},
                        )
                    )
            else:
                # Non-spatial relation positive.
                probes.append(
                    Probe(
                        ProbeType.RELATIONSHIP,
                        f"What is the {name} {rname}?",
                        tgt_name,
                        is_counterfactual=False,
                        source_facts={"subject": name, "relation": rname, "object": tgt_name},
                    )
                )
                # Compound grounded claim.
                probes.append(
                    Probe(
                        ProbeType.COMPOUND_CLAIM,
                        f'Is the statement "The {name} is {rname} the {tgt_name}" '
                        f"supported by the image?",
                        "yes",
                        is_counterfactual=False,
                        source_facts={"subject": name, "relation": rname, "object": tgt_name},
                    )
                )

    # --- Absent-object counterfactual (abstention-adjacent) ---
    for candidate in ("suitcase", "elephant", "airplane"):
        if candidate not in present_names:
            probes.append(
                Probe(
                    ProbeType.OBJECT,
                    f"Is there a {candidate} in the image?",
                    "no",
                    is_counterfactual=True,
                    source_facts={"absent_object": candidate},
                )
            )
            break

    return probes


def _attr_dimension(attr: str) -> str:
    """Guess the question dimension word for an attribute value."""
    colors = {"red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey"}
    materials = {"wooden", "metal", "plastic", "glass", "leather"}
    if attr in colors:
        return "color"
    if attr in materials:
        return "material"
    return "attribute"


def _counterfactual_attr(attr: str) -> str:
    """Return a plausible-but-wrong attribute of the same dimension."""
    colors = ["red", "blue", "green", "black", "white", "yellow"]
    for c in colors:
        if c != attr:
            return c if attr in colors else "red"
    return "red"


def _canonical_spatial(rname: str, eval_cfg: dict) -> str | None:
    """Map a surface spatial phrase to a canonical relation, else None."""
    mapping = {
        "to the left of": "left_of",
        "left of": "left_of",
        "to the right of": "right_of",
        "right of": "right_of",
        "above": "above",
        "below": "below",
        "under": "under",
        "on top of": "on",
        "in front of": "in_front_of",
        "behind": "behind",
        "inside": "inside",
        "near": "near",
    }
    return mapping.get(rname)


def _surface_spatial(canonical: str) -> str:
    surface = {
        "left_of": "to the left of",
        "right_of": "to the right of",
        "above": "above",
        "below": "below",
        "in_front_of": "in front of",
        "behind": "behind",
        "on": "on top of",
        "under": "under",
        "inside": "inside",
        "near": "near",
    }
    return surface.get(canonical, canonical.replace("_", " "))


__all__ = ["Probe", "ProbeType", "generate_probes"]

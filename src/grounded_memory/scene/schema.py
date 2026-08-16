"""Pydantic schema for the structured grounded scene representation.

Central design principle::

    observation != interpretation

A model may *observe* that a hand is touching a mug; concluding the person is
"drinking coffee" is an *interpretation* that may be plausible but unsupported.
Every factual claim therefore carries an explicit evidence type and confidence.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class EvidenceType(str, Enum):
    """How well a claim is grounded in directly observable visual evidence."""

    DIRECT = "direct_or_supported"       # visible in the image
    INFERRED = "inferred"                 # plausible interpretation, not directly shown
    UNSUPPORTED = "unsupported"           # asserted without visual support
    INSUFFICIENT = "insufficient_evidence"  # model declines: not determinable


class ConfidenceCategory(str, Enum):
    """Coarse confidence bucket, usable when a numeric score is unavailable."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Entity(BaseModel):
    """A detected object/agent in the scene."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable within-scene id, e.g. 'person_1'.")
    label: str = Field(..., description="Object/agent category, e.g. 'woman'.")
    attributes: list[str] = Field(default_factory=list, description="e.g. ['red', 'wooden'].")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A non-spatial relation between two entities (subject -> relation -> object)."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., description="Entity id.")
    relation: str = Field(..., description="e.g. 'holding', 'using', 'wearing'.")
    object: str = Field(..., description="Entity id or literal label.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SpatialRelation(BaseModel):
    """A spatial relation (left_of, above, on, ...) between two entities."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: str = Field(..., description="Canonical spatial relation.")
    object: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Claim(BaseModel):
    """A single factual assertion with its supporting evidence and type."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence: str = Field(default="", description="What in the image supports this claim.")
    evidence_type: EvidenceType = EvidenceType.INFERRED
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_category: ConfidenceCategory | None = None


class Uncertainty(BaseModel):
    """An explicitly flagged uncertainty or ambiguity in the scene."""

    model_config = ConfigDict(extra="forbid")

    about: str = Field(..., description="What is uncertain, e.g. 'brand of backpack'.")
    reason: str = Field(default="", description="Why it cannot be determined.")


class SceneRepresentation(BaseModel):
    """Full structured grounded representation produced by Condition C.

    ``raw_output`` and ``parse_ok`` let callers keep the original model text even
    when structured parsing fails (Reproducibility: never discard a prediction).
    """

    model_config = ConfigDict(extra="forbid")

    scene_summary: str = ""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    spatial_relations: list[SpatialRelation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    uncertainty: list[Uncertainty] = Field(default_factory=list)

    def object_labels(self) -> list[str]:
        """Convenience: all entity labels (lower-cased)."""
        return [e.label.lower() for e in self.entities]

    def supported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.evidence_type == EvidenceType.DIRECT]

    def unsupported_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.evidence_type == EvidenceType.UNSUPPORTED]


__all__ = [
    "ConfidenceCategory",
    "EvidenceType",
    "Entity",
    "Relationship",
    "SpatialRelation",
    "Claim",
    "Uncertainty",
    "SceneRepresentation",
]

"""Human-evaluation rating I/O for notebook 06's failure-analysis pass.

Mirrors ``data/review.py``'s ``ManifestRow``/``load_manifest``/``save_manifest``
pattern (full-file-rewrite-on-save CSV, so a rating session can be interrupted
and resumed without losing progress) for a different row shape: one row per
(image, model, prompt condition) rated on the 4-item binary rubric.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

RATING_CSV_COLUMNS = [
    "image_id", "primary_bucket", "model_id", "condition", "prediction_excerpt",
    "object_attribute_hallucination", "relation_spatial_error",
    "unsupported_action_intent_inference", "appropriate_abstention_hedging",
    "failure_taxonomy", "notes", "rater", "rated_at",
]


@dataclass
class RatingRow:
    """One human-rated (image, model, condition) item.

    ``appropriate_abstention_hedging`` is the one rubric item where True means
    the response is GOOD (it correctly hedged/abstained) -- opposite polarity
    from the other three, where True means a failure was observed. Kept as
    the user specified; surfaced explicitly in the notebook UI so a rater
    doesn't misread it.

    ``rated_at`` is empty until the row has actually been rated -- used to
    tell "queued but not yet rated" apart from "rated" when resuming.
    """

    image_id: str
    primary_bucket: str
    model_id: str
    condition: str
    prediction_excerpt: str
    object_attribute_hallucination: bool = False
    relation_spatial_error: bool = False
    unsupported_action_intent_inference: bool = False
    appropriate_abstention_hedging: bool = False
    failure_taxonomy: str = ""
    notes: str = ""
    rater: str = ""
    rated_at: str = ""

    def to_csv_row(self) -> list[str]:
        return [
            self.image_id, self.primary_bucket, self.model_id, self.condition,
            self.prediction_excerpt,
            str(self.object_attribute_hallucination), str(self.relation_spatial_error),
            str(self.unsupported_action_intent_inference),
            str(self.appropriate_abstention_hedging),
            self.failure_taxonomy, self.notes, self.rater, self.rated_at,
        ]

    @staticmethod
    def from_csv_dict(raw: dict[str, str]) -> RatingRow:
        def as_bool(s: str) -> bool:
            return s.strip().lower() == "true"

        return RatingRow(
            image_id=raw["image_id"],
            primary_bucket=raw["primary_bucket"],
            model_id=raw["model_id"],
            condition=raw["condition"],
            prediction_excerpt=raw["prediction_excerpt"],
            object_attribute_hallucination=as_bool(raw["object_attribute_hallucination"]),
            relation_spatial_error=as_bool(raw["relation_spatial_error"]),
            unsupported_action_intent_inference=as_bool(
                raw["unsupported_action_intent_inference"]
            ),
            appropriate_abstention_hedging=as_bool(raw["appropriate_abstention_hedging"]),
            failure_taxonomy=raw["failure_taxonomy"],
            notes=raw["notes"],
            rater=raw["rater"],
            rated_at=raw["rated_at"],
        )


def load_ratings(path: str | Path) -> list[RatingRow]:
    """Load a ratings CSV, in ``RATING_CSV_COLUMNS``'s exact column order."""
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return [RatingRow.from_csv_dict(raw) for raw in csv.DictReader(fh)]


def save_ratings(rows: list[RatingRow], path: str | Path) -> None:
    """Write rows to CSV using ``RATING_CSV_COLUMNS``'s exact column order."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(RATING_CSV_COLUMNS)
        for row in rows:
            writer.writerow(row.to_csv_row())


__all__ = ["RATING_CSV_COLUMNS", "RatingRow", "load_ratings", "save_ratings"]

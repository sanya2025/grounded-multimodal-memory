"""Temporal scoring and filtering utilities for memory retrieval."""

from __future__ import annotations

from datetime import datetime

from grounded_memory.memory.events import MemoryEvent


def recency_score(event: MemoryEvent, reference: datetime, half_life_s: float = 3600.0) -> float:
    """Exponential-decay recency score in [0, 1] relative to a reference time.

    Events at or after ``reference`` score 1.0; older events decay with the given
    half-life (default 1 hour).
    """
    dt = (reference - event.timestamp).total_seconds()
    if dt <= 0:
        return 1.0
    return float(0.5 ** (dt / half_life_s))


def within_window(
    event: MemoryEvent, start: datetime | None, end: datetime | None
) -> bool:
    """Whether an event falls within an optional [start, end] window (inclusive)."""
    if start is not None and event.timestamp < start:
        return False
    if end is not None and event.timestamp > end:
        return False
    return True


def temporal_relevance(
    event: MemoryEvent,
    reference: datetime | None,
    constraint: str = "recency",
    half_life_s: float = 3600.0,
) -> float:
    """Temporal relevance in [0, 1].

    ``constraint``:
      - "recency": closer to ``reference`` in the past scores higher.
      - "before":  events strictly before ``reference`` score by recency, else 0.
      - "after":   events at/after ``reference`` score 1, else 0.
      - "any":     constant 1.0.
    """
    if reference is None or constraint == "any":
        return 1.0
    if constraint == "before":
        return recency_score(event, reference, half_life_s) if event.timestamp < reference else 0.0
    if constraint == "after":
        return 1.0 if event.timestamp >= reference else 0.0
    return recency_score(event, reference, half_life_s)


def latest_event(events: list[MemoryEvent]) -> MemoryEvent | None:
    """Most recent event by timestamp, or None if empty."""
    return max(events, key=lambda e: e.timestamp) if events else None


__all__ = ["recency_score", "within_window", "temporal_relevance", "latest_event"]

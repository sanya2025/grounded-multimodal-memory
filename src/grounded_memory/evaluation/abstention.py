"""Abstention detection and metrics.

For probes that intentionally cannot be answered from the image, the desired
behavior is to abstain ("insufficient visual evidence"). We measure how often the
model correctly abstains versus fabricating a concrete answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from grounded_memory.config import load_config


@lru_cache(maxsize=1)
def _abstention_phrases() -> tuple[str, ...]:
    cfg = load_config("evaluation")
    return tuple(p.lower() for p in cfg.get("abstention", {}).get("phrases", []))


def is_abstention(response: str, phrases: tuple[str, ...] | None = None) -> bool:
    """True if the response signals insufficient evidence / declines to answer."""
    text = response.lower()
    for phrase in phrases or _abstention_phrases():
        if phrase in text:
            return True
    return False


@dataclass
class AbstentionMetrics:
    correct_abstention_rate: float  # of items requiring abstention, fraction abstained
    false_answer_rate: float        # of items requiring abstention, fraction answered
    n_requiring_abstention: int


def abstention_metrics(
    responses: list[str], requires_abstention: list[bool]
) -> AbstentionMetrics:
    """Compute correct-abstention and false-answer rates.

    ``requires_abstention[i]`` marks probes with no answerable evidence.
    """
    if len(responses) != len(requires_abstention):
        raise ValueError("responses and requires_abstention must be the same length")
    idx = [i for i, req in enumerate(requires_abstention) if req]
    n = len(idx)
    if n == 0:
        return AbstentionMetrics(0.0, 0.0, 0)
    abstained = sum(1 for i in idx if is_abstention(responses[i]))
    car = abstained / n
    return AbstentionMetrics(car, 1.0 - car, n)


__all__ = ["is_abstention", "AbstentionMetrics", "abstention_metrics"]

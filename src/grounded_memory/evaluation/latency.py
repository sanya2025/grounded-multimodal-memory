"""Latency aggregation and decomposition (Experiment 3).

Never report a single total-latency number without its components. These helpers
aggregate per-run ``component_latencies`` dicts into mean/median/percentiles.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class LatencyStats:
    mean: float
    median: float
    p90: float
    n: int


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def summarize(values: Sequence[float]) -> LatencyStats:
    vals = list(values)
    if not vals:
        return LatencyStats(0.0, 0.0, 0.0, 0)
    return LatencyStats(
        mean=statistics.fmean(vals),
        median=statistics.median(vals),
        p90=_percentile(vals, 0.90),
        n=len(vals),
    )


def decompose(runs: Sequence[dict[str, float]]) -> dict[str, LatencyStats]:
    """Aggregate a list of per-run component-latency dicts by component."""
    components: dict[str, list[float]] = {}
    for run in runs:
        for name, value in run.items():
            components.setdefault(name, []).append(float(value))
    return {name: summarize(vals) for name, vals in components.items()}


__all__ = ["LatencyStats", "summarize", "decompose"]

"""Bootstrap confidence intervals and paired bootstrap comparisons.

Used to avoid overstating small differences between prompt conditions or methods.
Seeded for reproducibility (do not use global RNG state).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class CI:
    point: float
    low: float
    high: float


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20260815,
) -> CI:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return CI(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return CI(float(arr.mean()), float(low), float(high))


@dataclass
class PairedResult:
    mean_diff: float
    low: float
    high: float
    prob_positive: float  # fraction of bootstrap resamples where a > b


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 20260815,
) -> PairedResult:
    """Paired bootstrap of (a - b) over per-item scores on the SAME items.

    Use when both methods ran on the same images/questions.
    """
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.shape != bv.shape:
        raise ValueError("paired_bootstrap requires equal-length, aligned inputs")
    if av.size == 0:
        return PairedResult(0.0, 0.0, 0.0, 0.5)
    diff = av - bv
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_means = diff[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return PairedResult(
        mean_diff=float(diff.mean()),
        low=float(low),
        high=float(high),
        prob_positive=float((boot_means > 0).mean()),
    )


__all__ = ["CI", "bootstrap_ci", "PairedResult", "paired_bootstrap"]

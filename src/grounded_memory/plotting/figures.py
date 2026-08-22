"""Publication-quality figure helpers.

All figures are generated from frozen result files by scripts/notebooks (never
hand-edited). matplotlib is imported lazily so the base package stays light.
Figures use a single, colorblind-safe categorical palette for consistency.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Colorblind-safe qualitative palette (Okabe-Ito).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def _import_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'viz' extra: pip install -e '.[viz]'") from exc


def grouped_bar(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    ylabel: str,
    title: str,
    out_path: str | Path,
) -> Path:
    """Grouped bar chart (e.g. metric by prompt condition per model)."""
    plt = _import_mpl()
    import numpy as np

    x = np.arange(len(categories))
    n = len(series)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, values) in enumerate(series.items()):
        ax.bar(x + i * width, values, width, label=label, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def accuracy_latency_scatter(
    points: list[dict],
    out_path: str | Path,
    title: str = "Accuracy vs Latency (marker size = memory footprint)",
) -> Path:
    """The E3.5 tradeoff scatter: accuracy (y) vs latency (x), size = memory."""
    plt = _import_mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, p in enumerate(points):
        ax.scatter(
            p["latency_s"], p["accuracy"],
            s=max(40.0, p.get("memory_gb", 1.0) * 60.0),
            color=PALETTE[i % len(PALETTE)], alpha=0.8, edgecolors="black",
        )
        ax.annotate(p.get("label", ""), (p["latency_s"], p["accuracy"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


__all__ = ["PALETTE", "grouped_bar", "accuracy_latency_scatter"]

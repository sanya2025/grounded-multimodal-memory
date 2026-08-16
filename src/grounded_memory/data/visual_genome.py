"""Visual Genome loader (supporting annotation source only).

Not a second full benchmark. Use for richer original annotations (regions,
bounding boxes, dense relationships) when a GQA scene graph is too sparse.
See https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Thin JSON loader with a clear error if the VG dump is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Visual Genome file not found: {p}. Download VG and set "
            f"configs/datasets.yaml -> visual_genome.root. See REFERENCES.md."
        )
    return json.loads(p.read_text())


# TODO: add region-graph / bounding-box accessors as needed by failure analysis.
__all__ = ["load_json"]

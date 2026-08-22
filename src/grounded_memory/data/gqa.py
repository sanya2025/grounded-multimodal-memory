"""GQA loader utilities.

GQA is NOT bundled. Download it separately (see scripts/download_gqa.py and
REFERENCES.md) and point configs/datasets.yaml at it. These helpers read the
scene-graph JSON and expose a uniform ``SceneGraph`` view used by selection and
probe generation.

GQA scene-graph JSON shape (per image id)::

    {
      "2407890": {
        "width": 640, "height": 480,
        "objects": {
          "o1": {"name": "umbrella", "attributes": ["black"],
                 "relations": [{"name": "to the left of", "object": "o2"}],
                 "x": .., "y": .., "w": .., "h": ..},
          ...
        }
      }
    }
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grounded_memory.config import load_config


@dataclass
class SceneGraph:
    image_id: str
    raw: dict[str, Any]

    @property
    def objects(self) -> dict[str, Any]:
        return self.raw.get("objects", {})

    def num_objects(self) -> int:
        return len(self.objects)

    def num_relations(self) -> int:
        return sum(len(o.get("relations", [])) for o in self.objects.values())

    def num_attributes(self) -> int:
        return sum(len(o.get("attributes", [])) for o in self.objects.values())

    def object_names(self) -> list[str]:
        return [o.get("name", "") for o in self.objects.values() if o.get("name")]


def scene_graph_path(datasets_cfg: dict | None = None, split: str | None = None) -> Path:
    cfg = datasets_cfg or load_config("datasets")
    gqa = cfg["gqa"]
    split = split or gqa.get("benchmark_split", "val")
    return Path(gqa["root"]) / gqa["scene_graphs"][split]


def load_scene_graphs(path: str | Path) -> dict[str, SceneGraph]:
    """Load a GQA scene-graph JSON file into ``{image_id: SceneGraph}``."""
    data = json.loads(Path(path).read_text())
    return {img_id: SceneGraph(img_id, sg) for img_id, sg in data.items()}


def iter_scene_graphs(path: str | Path) -> Iterator[SceneGraph]:
    """Stream scene graphs (memory-friendly for the full GQA file)."""
    data = json.loads(Path(path).read_text())
    for img_id, sg in data.items():
        yield SceneGraph(img_id, sg)


__all__ = ["SceneGraph", "scene_graph_path", "load_scene_graphs", "iter_scene_graphs"]

#!/usr/bin/env python
"""Print instructions (and optionally fetch) the GQA dataset.

We deliberately do NOT auto-download multi-GB data on import. Run this script to
see the exact steps and target paths. Pass --urls to print the official download
links only.

GQA: https://cs.stanford.edu/people/dorarad/gqa/about.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grounded_memory.config import load_config

INSTRUCTIONS = """\
GQA download steps
==================
1. Visit the official page and accept terms:
   https://cs.stanford.edu/people/dorarad/gqa/download.html

2. Download and unzip into your GQA root (configs/datasets.yaml -> gqa.root):
   - Images        -> {root}/images/
   - Scene graphs  -> {root}/sceneGraphs/ (train_sceneGraphs.json, val_sceneGraphs.json)
   - Questions     -> {root}/questions/   (val_balanced_questions.json)

3. Verify with:
   python -c "from grounded_memory.data.gqa import scene_graph_path, load_scene_graphs; \\
              p=scene_graph_path(split='val'); print('reading', p); \\
              sg=load_scene_graphs(p); print(len(sg), 'scene graphs')"

Do NOT commit raw images or JSON to git (see .gitignore). Only the frozen
120-image manifest (data/manifests/*.csv) is versioned.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="GQA download helper (prints instructions).")
    ap.add_argument("--urls", action="store_true", help="Print official URLs only.")
    args = ap.parse_args()

    cfg = load_config("datasets")
    root = Path(cfg["gqa"]["root"])
    if args.urls:
        print("https://cs.stanford.edu/people/dorarad/gqa/download.html")
        print("https://arxiv.org/abs/1902.09506")
        return
    print(INSTRUCTIONS.format(root=root))
    print(f"Configured GQA root: {root.resolve()}")


if __name__ == "__main__":
    main()

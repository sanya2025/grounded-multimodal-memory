"""Dataset loaders (GQA, Visual Genome, CLEVR) and manifest selection."""

from __future__ import annotations

from grounded_memory.data.selection import (
    BUCKETS,
    ManifestRow,
    bucket_features,
    build_candidate_manifest,
)

__all__ = ["BUCKETS", "ManifestRow", "build_candidate_manifest", "bucket_features"]

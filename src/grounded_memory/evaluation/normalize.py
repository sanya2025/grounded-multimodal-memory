"""Label / relation normalization driven by configs/evaluation.yaml."""

from __future__ import annotations

from functools import lru_cache

from grounded_memory.config import load_config


@lru_cache(maxsize=1)
def _eval_cfg() -> dict:
    return load_config("evaluation")


def normalize_object(label: str, cfg: dict | None = None) -> str:
    """Lower-case, strip a trailing plural 's', and map synonyms to a canonical."""
    cfg = cfg or _eval_cfg()
    norm_cfg = cfg.get("object_normalization", {})
    s = label.strip().lower() if norm_cfg.get("lowercase", True) else label.strip()
    if norm_cfg.get("strip_plurals", True) and len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    for canonical, synonyms in norm_cfg.get("synonyms", {}).items():
        syns = {normalize_simple(x) for x in synonyms}
        if s in syns or s == canonical:
            return canonical
    return s


def normalize_simple(label: str) -> str:
    s = label.strip().lower()
    if len(s) > 3 and s.endswith("s"):
        s = s[:-1]
    return s


def normalize_relation(relation: str, cfg: dict | None = None) -> str:
    """Map a surface relation phrase to its canonical form."""
    cfg = cfg or _eval_cfg()
    r = relation.strip().lower()
    for canonical, surface_forms in cfg.get("relation_normalization", {}).items():
        if r == canonical or r in {s.lower() for s in surface_forms}:
            return canonical
    return r


__all__ = ["normalize_object", "normalize_relation", "normalize_simple"]

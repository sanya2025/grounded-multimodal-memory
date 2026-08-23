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


_AUX_PREFIXES = ("is ", "are ", "was ", "were ")


def normalize_relation(relation: str, cfg: dict | None = None) -> str:
    """Map a surface relation phrase to its canonical form.

    Strips a leading auxiliary verb ("is"/"are"/"was"/"were") before matching,
    so a model's natural phrasing ("is above") matches the same vocabulary
    entry as the bare form ("above") without needing every "is X" variant
    enumerated in configs/evaluation.yaml.
    """
    cfg = cfg or _eval_cfg()
    r = relation.strip().lower()
    for prefix in _AUX_PREFIXES:
        if r.startswith(prefix):
            r = r[len(prefix):]
            break
    for canonical, surface_forms in cfg.get("relation_normalization", {}).items():
        if r == canonical or r in {s.lower() for s in surface_forms}:
            return canonical
    return r


def attribute_category(attribute: str, cfg: dict | None = None) -> str | None:
    """Which mutually-exclusive category (e.g. "color") an attribute belongs to.

    Used to detect CONTRADICTED attribute claims: an object annotated "blue"
    but predicted "red" is a conflict because color is mutually exclusive.
    Returns None for attributes with no configured category (most attributes
    stay NOT_VERIFIABLE rather than risk a false CONTRADICTED).
    """
    cfg = cfg or _eval_cfg()
    a = normalize_simple(attribute)
    for category, members in cfg.get("attribute_categories", {}).items():
        if a in {normalize_simple(m) for m in members}:
            return category
    return None


def spatial_opposite(relation: str, cfg: dict | None = None) -> str | None:
    """The opposite of a canonical spatial relation (e.g. "left_of" -> "right_of").

    Looks up the ALREADY-NORMALIZED relation in configs/evaluation.yaml's
    spatial_relations.opposites (bidirectional -- both directions are listed
    explicitly there, so no reverse lookup is needed here).
    """
    cfg = cfg or _eval_cfg()
    return cfg.get("spatial_relations", {}).get("opposites", {}).get(relation)


__all__ = [
    "normalize_object",
    "normalize_relation",
    "normalize_simple",
    "attribute_category",
    "spatial_opposite",
]

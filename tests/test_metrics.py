"""Tests for evaluation metrics: objects, hallucination, abstention, QA, stats."""

from __future__ import annotations

import math

from grounded_memory.evaluation.abstention import abstention_metrics, is_abstention
from grounded_memory.evaluation.categorical import tuple_prf1
from grounded_memory.evaluation.grounding import (
    EvidenceLabel,
    verify_attribute_claim,
    verify_relation_claim,
)
from grounded_memory.evaluation.hallucination import (
    evidence_support_rate,
    hallucination_rate,
)
from grounded_memory.evaluation.normalize import (
    attribute_category,
    normalize_relation,
    spatial_opposite,
)
from grounded_memory.evaluation.objects import object_prf1
from grounded_memory.evaluation.qa import normalize_answer, qa_accuracy
from grounded_memory.evaluation.stats import bootstrap_ci, paired_bootstrap


def test_object_prf1_basic():
    r = object_prf1(predicted=["dog", "cat", "car"], reference=["dog", "cat", "tree"])
    assert r.tp == 2 and r.fp == 1 and r.fn == 1
    assert math.isclose(r.precision, 2 / 3)
    assert math.isclose(r.recall, 2 / 3)
    assert math.isclose(r.f1, 2 / 3)


def test_object_prf1_normalizes_plurals_and_synonyms():
    r = object_prf1(predicted=["dogs", "man"], reference=["dog", "person"])
    # "dogs"->"dog"; "man"->"person" via synonyms.
    assert r.recall == 1.0


def test_hallucination_rate_strict_ignores_not_verifiable():
    labels = [
        EvidenceLabel.SUPPORTED,
        EvidenceLabel.CONTRADICTED,
        EvidenceLabel.NOT_VERIFIABLE,
        EvidenceLabel.NOT_VERIFIABLE,
    ]
    # strict: 1 contradicted / (1 supported + 1 contradicted) = 0.5
    assert math.isclose(hallucination_rate(labels, strict=True), 0.5)
    # non-strict upper bound: (1 + 2) / 4 = 0.75
    assert math.isclose(hallucination_rate(labels, strict=False), 0.75)
    assert math.isclose(evidence_support_rate(labels), 0.5)


def test_is_abstention_detects_phrases():
    assert is_abstention("The available visual evidence is insufficient.")
    assert not is_abstention("It is a North Face backpack.")


def test_abstention_metrics():
    responses = [
        "It is red.",                                   # required abstention, answered -> false
        "The brand cannot be determined.",              # required abstention, abstained -> correct
        "There is a dog.",                              # not required
    ]
    requires = [True, True, False]
    m = abstention_metrics(responses, requires)
    assert m.n_requiring_abstention == 2
    assert math.isclose(m.correct_abstention_rate, 0.5)
    assert math.isclose(m.false_answer_rate, 0.5)


def test_qa_accuracy_and_normalization():
    assert normalize_answer("The Dog.") == "dog"
    assert normalize_answer("Yes!") == "yes"
    res = qa_accuracy(
        predictions=["The keys are in the backpack.", "blue", "left"],
        references=["backpack", "red", "left"],
    )
    assert res.correct == 2
    assert res.total == 3


def test_tuple_prf1_reports_precision_and_recall_separately():
    predicted = {("person", "holding", "cup"), ("person", "wearing", "hat")}
    reference = {("person", "holding", "cup"), ("person", "riding", "bike")}
    r = tuple_prf1(predicted, reference)
    assert r.tp == 1 and r.fp == 1 and r.fn == 1
    assert math.isclose(r.precision, 0.5)
    assert math.isclose(r.recall, 0.5)


def test_tuple_prf1_empty_reference_does_not_crash():
    r = tuple_prf1(predicted={("a", "b", "c")}, reference=set())
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0


def test_normalize_relation_strips_auxiliary_verb_prefix():
    # A model's natural "is above" must match the bare canonical "above"
    # without every "is X" variant being enumerated in configs/evaluation.yaml.
    assert normalize_relation("is above") == normalize_relation("above") == "above"
    assert normalize_relation("is holding") == "holding"


def test_normalize_relation_maps_gqa_positional_phrases_to_canonical():
    assert normalize_relation("to the left of") == "left_of"
    assert normalize_relation("to the right of") == "right_of"
    assert normalize_relation("in front of") == "in_front_of"


def test_attribute_category_groups_colors_and_materials():
    assert attribute_category("red") == "color"
    assert attribute_category("blue") == "color"
    assert attribute_category("wood") == "material"
    assert attribute_category("tall") is None  # size deliberately excluded


def test_spatial_opposite_bidirectional():
    assert spatial_opposite("left_of") == "right_of"
    assert spatial_opposite("right_of") == "left_of"
    assert spatial_opposite("holding") is None  # non-spatial, no configured opposite


def _sg(objects: dict) -> dict:
    return {"objects": objects}


def test_verify_attribute_claim_contradicted_on_conflicting_color():
    sg = _sg({"o1": {"name": "car", "attributes": ["blue"], "relations": []}})
    assert verify_attribute_claim("car", "red", sg) == EvidenceLabel.CONTRADICTED
    assert verify_attribute_claim("car", "blue", sg) == EvidenceLabel.SUPPORTED


def test_verify_attribute_claim_not_verifiable_when_no_category_conflict():
    sg = _sg({"o1": {"name": "car", "attributes": ["blue"], "relations": []}})
    # "shiny" has no configured category, so it can't be judged CONTRADICTED
    # even though it's not the annotated attribute -- absence isn't proof.
    assert verify_attribute_claim("car", "shiny", sg) == EvidenceLabel.NOT_VERIFIABLE


def test_verify_attribute_claim_size_words_never_contradict():
    # "tall" and "large" are excluded from attribute_categories on purpose --
    # an object can plausibly be both, so they must never show CONTRADICTED.
    sg = _sg({"o1": {"name": "man", "attributes": ["tall"], "relations": []}})
    assert verify_attribute_claim("man", "large", sg) == EvidenceLabel.NOT_VERIFIABLE


def test_verify_attribute_claim_object_not_annotated():
    sg = _sg({"o1": {"name": "car", "attributes": ["blue"], "relations": []}})
    assert verify_attribute_claim("dog", "brown", sg) == EvidenceLabel.NOT_VERIFIABLE


def test_verify_relation_claim_contradicted_on_spatial_opposite():
    sg = _sg({
        "o1": {"name": "cup", "attributes": [], "relations": [{"name": "left_of", "object": "o2"}]},
        "o2": {"name": "plate", "attributes": [], "relations": []},
    })
    assert verify_relation_claim("cup", "right_of", "plate", sg) == EvidenceLabel.CONTRADICTED
    assert verify_relation_claim("cup", "left_of", "plate", sg) == EvidenceLabel.SUPPORTED


def test_verify_relation_claim_non_spatial_mismatch_stays_not_verifiable():
    # "holding" has no configured opposite, so a wrong non-spatial claim
    # can't be detected as CONTRADICTED -- stays the conservative default.
    sg = _sg({
        "o1": {
            "name": "person", "attributes": [],
            "relations": [{"name": "holding", "object": "o2"}],
        },
        "o2": {"name": "cup", "attributes": [], "relations": []},
    })
    assert verify_relation_claim("person", "wearing", "cup", sg) == EvidenceLabel.NOT_VERIFIABLE


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    ci1 = bootstrap_ci(vals, n_boot=2000, seed=42)
    ci2 = bootstrap_ci(vals, n_boot=2000, seed=42)
    assert ci1 == ci2  # reproducible
    assert ci1.low <= ci1.point <= ci1.high


def test_paired_bootstrap_detects_improvement():
    grounded = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    standard = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    res = paired_bootstrap(grounded, standard, n_boot=2000, seed=7)
    assert res.mean_diff > 0
    assert res.prob_positive > 0.9

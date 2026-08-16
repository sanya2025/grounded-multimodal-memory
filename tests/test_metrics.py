"""Tests for evaluation metrics: objects, hallucination, abstention, QA, stats."""

from __future__ import annotations

import math

from grounded_memory.evaluation.abstention import abstention_metrics, is_abstention
from grounded_memory.evaluation.grounding import EvidenceLabel
from grounded_memory.evaluation.hallucination import (
    evidence_support_rate,
    hallucination_rate,
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

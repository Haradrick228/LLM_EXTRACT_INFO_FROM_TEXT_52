"""Token-overlap F1: симметрия, пустые случаи, порог."""
import math

import pytest

from core.rag.metrics.token_overlap_f1_hit import (
    make_token_overlap_f1_hit,
    token_overlap_f1,
    token_overlap_f1_hit,
)


def test_identical_strings_score_one():
    assert math.isclose(token_overlap_f1("Критерии успеха", "критерии успеха"), 1.0)


def test_disjoint_strings_score_zero():
    assert token_overlap_f1("одно", "другое") == 0.0


def test_partial_overlap_between_0_and_1():
    s = token_overlap_f1("дифференциация заданий по уровню сложности", "уровню сложности")
    assert 0.0 < s < 1.0


def test_punctuation_ignored():
    a = "критерии, успеха!"
    b = "критерии успеха"
    assert token_overlap_f1(a, b) >= 0.99


def test_threshold_default():
    assert token_overlap_f1_hit("критерии успеха", "критерии успеха")
    assert not token_overlap_f1_hit("ничего общего", "критерии успеха")


def test_custom_threshold_factory():
    strict = make_token_overlap_f1_hit(threshold=0.99)
    assert strict("одно слово два", "одно слово два")
    assert not strict("одно слово два", "одно слово")


def test_empty_inputs():
    assert token_overlap_f1("", "") == 0.0
    assert token_overlap_f1("слово", "") == 0.0
    assert token_overlap_f1("", "слово") == 0.0


@pytest.mark.parametrize(
    "a,b",
    [
        ("alpha beta gamma", "gamma beta alpha"),
        ("ёлка лесом росла", "ЁЛКА лесом РОСЛА"),
    ],
)
def test_symmetry_and_case(a: str, b: str):
    assert math.isclose(token_overlap_f1(a, b), token_overlap_f1(b, a))
    assert token_overlap_f1(a, b) >= 0.99

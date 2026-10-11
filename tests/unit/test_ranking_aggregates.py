"""recall@k / precision@k / MRR / nDCG@k поверх pluggable hit-fn."""
import math

from core.rag.metrics.ranking_aggregates import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from core.rag.metrics.substring_match_hit import substring_match_hit
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit


def test_recall_at_k_substring_perfect_hit():
    docs = ["foo", "bar", "критерии успеха в задании"]
    refs = ["критерии успеха"]
    assert recall_at_k(docs, refs, 3, hit_fn=substring_match_hit) == 1.0
    assert recall_at_k(docs, refs, 2, hit_fn=substring_match_hit) == 0.0


def test_recall_at_k_no_refs_returns_zero():
    assert recall_at_k(["a", "b"], [], 5, hit_fn=substring_match_hit) == 0.0


def test_precision_at_k_partial():
    docs = ["a слово", "irrelevant", "слово"]
    refs = ["слово"]
    p = precision_at_k(docs, refs, 3, hit_fn=substring_match_hit)
    assert math.isclose(p, 2.0 / 3.0)


def test_mrr_first_position():
    assert mean_reciprocal_rank(["слово"], ["слово"], hit_fn=substring_match_hit) == 1.0


def test_mrr_third_position():
    docs = ["a", "b", "слово"]
    refs = ["слово"]
    assert math.isclose(mean_reciprocal_rank(docs, refs, hit_fn=substring_match_hit), 1.0 / 3.0)


def test_mrr_no_hit_zero():
    assert mean_reciprocal_rank(["a", "b"], ["c"], hit_fn=substring_match_hit) == 0.0


def test_ndcg_clipped_at_1():
    docs = ["слово", "слово", "слово"]
    refs = ["слово"]
    val = ndcg_at_k(docs, refs, 3, hit_fn=substring_match_hit)
    assert 0.0 < val <= 1.0


def test_pluggable_hit_fn_token_f1():
    docs = ["критерии успеха в задании", "irrelevant", "по сложности задания"]
    refs = ["критерии успеха"]
    assert recall_at_k(docs, refs, 1, hit_fn=token_overlap_f1_hit) == 1.0
    assert recall_at_k(docs, refs, 1, hit_fn=substring_match_hit) == 1.0

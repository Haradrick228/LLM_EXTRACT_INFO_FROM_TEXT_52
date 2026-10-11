"""Regression: nDCG must not exceed 1.0 (multi-rank matches same ref)."""
from eval.retrieval.run_retrieval_eval import _refs_for_retrieval_metrics, ndcg


def test_refs_for_retrieval_metrics_prefers_references():
    row = {"references": ["a", "b"], "reference_quote": "ignored if refs"}
    assert _refs_for_retrieval_metrics(row) == ["a", "b"]


def test_refs_for_retrieval_metrics_fallback_reference_quote():
    row = {"reference_quote": "  only quote  "}
    assert _refs_for_retrieval_metrics(row) == ["only quote"]


def test_refs_for_retrieval_metrics_empty():
    assert _refs_for_retrieval_metrics({}) == []


def test_ndcg_capped_when_multiple_docs_match_same_ref():
    ref = "unique substring ref"
    found = [ref + " a", ref + " b", ref + " c"]
    assert ndcg(found, [ref], 3) == 1.0


def test_ndcg_normal_single_hit():
    found = ["irrelevant", "good hit", "noise"]
    assert ndcg(found, ["good hit"], 3) < 1.0

"""HybridLinearWeightedRetriever: min-max нормализация + взвешенная сумма."""
from __future__ import annotations

from typing import List

import pytest

from core.rag.retrievers.hybrid_linear_weighted_retriever import (
    HybridLinearWeightedRetriever,
)
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)


class _StubRetriever:
    def __init__(self, name: str, docs: List[RetrievedDoc]) -> None:
        self.name = name
        self._docs = docs

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(docs=self._docs)


def _mk(text: str, rank: int, score: float, *, cid: str | None = None) -> RetrievedDoc:
    return RetrievedDoc(text=text, score=score, rank=rank, chunk_id=cid)


def test_weights_change_ranking_order():
    dense = _StubRetriever(
        "dense",
        [_mk("a", 1, 1.0, cid="A"), _mk("b", 2, 0.0, cid="B")],
    )
    bm25 = _StubRetriever(
        "bm25",
        [_mk("b", 1, 1.0, cid="B"), _mk("a", 2, 0.0, cid="A")],
    )
    # Equal weights — tie между A и B; берём дет. порядок.
    hlw_eq = HybridLinearWeightedRetriever([dense, bm25], weights={"dense": 1.0, "bm25": 1.0})
    res_eq = hlw_eq.search(RetrievalRequest(query="q", top_k=2))
    assert {d.chunk_id for d in res_eq.docs} == {"A", "B"}

    # Bias to BM25 — B первой.
    hlw_b = HybridLinearWeightedRetriever([dense, bm25], weights={"dense": 0.1, "bm25": 10.0})
    res_b = hlw_b.search(RetrievalRequest(query="q", top_k=1))
    assert res_b.docs[0].chunk_id == "B"


def test_minmax_handles_single_doc_per_retriever():
    a = _StubRetriever("dense", [_mk("a", 1, 0.5, cid="A")])
    b = _StubRetriever("bm25",  [_mk("b", 1, 0.7, cid="B")])
    hlw = HybridLinearWeightedRetriever([a, b])
    res = hlw.search(RetrievalRequest(query="q", top_k=2))
    # один документ в выдаче → min == max → нормализованный скор = 0.0 у обоих;
    # порядок детерминирован Pythonовским sorted (stable).
    assert {d.chunk_id for d in res.docs} == {"A", "B"}
    for d in res.docs:
        assert d.score == pytest.approx(0.0)


def test_records_source_ranks_metadata():
    a = _StubRetriever("dense", [_mk("a", 1, 1.0, cid="A")])
    b = _StubRetriever("bm25",  [_mk("a", 3, 0.4, cid="A")])
    hlw = HybridLinearWeightedRetriever([a, b])
    out = hlw.search(RetrievalRequest(query="q", top_k=1))
    md = out.docs[0].metadata["linear_weighted_source_ranks"]
    assert md == {"dense": 1, "bm25": 3}

"""HybridRrfRetriever: RRF score = sum 1/(k+rank), dedup по chunk_id, top_k."""
from __future__ import annotations

from typing import List

import pytest

from core.rag.retrievers.hybrid_rrf_retriever import HybridRrfRetriever
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
        # ignore top_k cap for simplicity
        return RetrievalResult(docs=self._docs)


def _mk(text: str, rank: int, *, cid: str | None = None, src: str | None = None) -> RetrievedDoc:
    return RetrievedDoc(text=text, score=1.0 / rank, rank=rank, chunk_id=cid, source=src)


def test_rrf_combines_two_retrievers_with_unique_chunks():
    a = _StubRetriever("dense", [_mk("a", 1, cid="A"), _mk("b", 2, cid="B")])
    b = _StubRetriever("bm25",  [_mk("c", 1, cid="C"), _mk("a", 2, cid="A")])
    rrf = HybridRrfRetriever([a, b], k=60)
    out = rrf.search(RetrievalRequest(query="q", top_k=4))
    keys = [d.chunk_id for d in out.docs]
    # A is in both → highest RRF score
    assert keys[0] == "A"
    assert set(keys) == {"A", "B", "C"}


def test_rrf_score_formula():
    a = _StubRetriever("dense", [_mk("a", 1, cid="A")])
    b = _StubRetriever("bm25",  [_mk("a", 2, cid="A")])
    rrf = HybridRrfRetriever([a, b], k=60)
    out = rrf.search(RetrievalRequest(query="q", top_k=1))
    assert out.docs[0].chunk_id == "A"
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert out.docs[0].score == pytest.approx(expected, rel=1e-9)


def test_rrf_dedup_by_text_when_no_id():
    a = _StubRetriever("dense", [_mk("same text", 1)])
    b = _StubRetriever("bm25",  [_mk("same text", 5)])
    rrf = HybridRrfRetriever([a, b], k=60)
    out = rrf.search(RetrievalRequest(query="q", top_k=4))
    assert len(out.docs) == 1


def test_rrf_top_k_respected():
    a = _StubRetriever(
        "dense", [_mk(f"a{i}", i + 1, cid=f"A{i}") for i in range(10)],
    )
    rrf = HybridRrfRetriever([a], k=60)
    out = rrf.search(RetrievalRequest(query="q", top_k=3))
    assert len(out.docs) == 3
    assert [d.rank for d in out.docs] == [1, 2, 3]


def test_rrf_records_source_ranks_per_retriever():
    a = _StubRetriever("dense", [_mk("a", 1, cid="A"), _mk("b", 2, cid="B")])
    b = _StubRetriever("bm25",  [_mk("a", 4, cid="A")])
    rrf = HybridRrfRetriever([a, b], k=60)
    out = rrf.search(RetrievalRequest(query="q", top_k=2))
    top = next(d for d in out.docs if d.chunk_id == "A")
    assert top.metadata["rrf_source_ranks"]["dense"] == 1
    assert top.metadata["rrf_source_ranks"]["bm25"] == 4

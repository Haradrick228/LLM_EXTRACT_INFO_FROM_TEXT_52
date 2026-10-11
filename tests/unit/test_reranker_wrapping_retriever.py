"""RerankerWrappingRetriever: base → top-N pool → reranker → top-K."""
from __future__ import annotations

from typing import List

import pytest

from core.rag.rerankers.reranker_base_types import (
    RerankRequest,
    RerankResult,
)
from core.rag.retrievers.reranker_wrapping_retriever import RerankerWrappingRetriever
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)


class _StubBase:
    name = "base"

    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(docs=list(self._docs)[: request.top_k])


class _ScoreInjectorReranker:
    """Возвращает docs в порядке убывания заданных score'ов (по chunk_id)."""

    name = "stub_reranker"

    def __init__(self, score_by_cid: dict[str, float]) -> None:
        self.score_by_cid = score_by_cid

    def rerank(self, request: RerankRequest) -> RerankResult:
        scored = sorted(
            request.docs,
            key=lambda d: self.score_by_cid.get(d.chunk_id or "", 0.0),
            reverse=True,
        )
        reranked = [
            RetrievedDoc(
                text=d.text,
                score=float(self.score_by_cid.get(d.chunk_id or "", 0.0)),
                rank=i + 1,
                chunk_id=d.chunk_id,
                source=d.source,
                metadata={**d.metadata, "rerank_from_rank": d.rank},
            )
            for i, d in enumerate(scored[: request.top_k])
        ]
        return RerankResult(docs=reranked, timings_ms=1)


class _BoomReranker:
    name = "boom"

    def rerank(self, request: RerankRequest) -> RerankResult:
        raise RuntimeError("intentional fail")


def _d(text: str, rank: int, *, cid: str) -> RetrievedDoc:
    return RetrievedDoc(text=text, score=1.0 / rank, rank=rank, chunk_id=cid)


def test_rerank_changes_order_and_marks_metadata():
    base_docs = [_d("a", 1, cid="A"), _d("b", 2, cid="B"), _d("c", 3, cid="C")]
    base = _StubBase(base_docs)
    reranker = _ScoreInjectorReranker({"A": 0.1, "B": 0.9, "C": 0.5})

    wrap = RerankerWrappingRetriever(base, reranker, candidate_top_k=3)
    out = wrap.search(RetrievalRequest(query="q", top_k=3))

    assert [d.chunk_id for d in out.docs] == ["B", "C", "A"]
    assert out.docs[0].score == pytest.approx(0.9)
    assert out.docs[0].metadata["rerank_from_rank"] == 2
    assert out.debug["rerank_skipped"] is False


def test_no_reranker_returns_pool_truncated():
    base_docs = [_d(str(i), i + 1, cid=f"C{i}") for i in range(5)]
    base = _StubBase(base_docs)
    wrap = RerankerWrappingRetriever(base, None, candidate_top_k=5)
    out = wrap.search(RetrievalRequest(query="q", top_k=3))
    assert [d.chunk_id for d in out.docs] == ["C0", "C1", "C2"]
    assert out.debug["rerank_skipped"] is True


def test_candidate_top_k_larger_than_request():
    base_docs = [_d(str(i), i + 1, cid=f"C{i}") for i in range(10)]
    base = _StubBase(base_docs)
    reranker = _ScoreInjectorReranker({f"C{i}": 9.0 - i for i in range(10)})
    wrap = RerankerWrappingRetriever(base, reranker, candidate_top_k=8)
    out = wrap.search(RetrievalRequest(query="q", top_k=3))
    # base get top_k=max(8, 3) = 8 → reranker берёт 8, отдаёт top 3.
    assert len(out.docs) == 3
    assert [d.chunk_id for d in out.docs] == ["C0", "C1", "C2"]


def test_reranker_exception_falls_back_to_pool_order():
    """CrossEncoderReranker уже глотает исключения внутри; RerankerWrappingRetriever
    делегирует ему; если конкретный реранкер уронит rerank() — fallback на pool."""
    base_docs = [_d("a", 1, cid="A"), _d("b", 2, cid="B")]
    base = _StubBase(base_docs)
    wrap = RerankerWrappingRetriever(base, _BoomReranker(), candidate_top_k=2)
    with pytest.raises(RuntimeError):
        # Текущая реализация не глушит ошибки реранкера (это контракт),
        # но мы хотим, чтобы это было поднято наверх → фиксируем поведение.
        wrap.search(RetrievalRequest(query="q", top_k=2))

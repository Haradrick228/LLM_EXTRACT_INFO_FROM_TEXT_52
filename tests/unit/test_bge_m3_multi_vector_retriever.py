"""BgeM3MultiVectorRetriever: late interaction + weighted fusion."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytest.importorskip("numpy")
import numpy as np

from core.rag.retrievers.bge_m3_multi_vector_retriever import (
    BgeM3MultiVectorRetriever,
    bgem3_colbert_score,
    bgem3_dense_score,
    bgem3_sparse_score,
)
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)


class _PoolRetriever:
    name = "pool"

    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult(docs=list(self._docs)[: request.top_k])


class _CannedEncoder:
    def __init__(self, q_pack: Dict[str, Any], d_pack: Dict[str, Any]) -> None:
        self._q = q_pack
        self._d = d_pack

    def encode_queries(self, queries: List[str]) -> Dict[str, Any]:
        return self._q

    def encode_documents(self, documents: List[str]) -> Dict[str, Any]:
        return self._d


def test_dense_score_cosine_unit_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert bgem3_dense_score(a, b) == pytest.approx(1.0)
    assert bgem3_dense_score(a, c) == pytest.approx(0.0)
    assert bgem3_dense_score([], a) == pytest.approx(0.0)


def test_sparse_score_dot_on_intersection():
    q = {"1": 0.5, "5": 0.3, "9": 0.1}
    d = {1: 1.0, "9": 2.0, 99: 0.4}  # ключи могут быть int/str
    expected = 0.5 * 1.0 + 0.1 * 2.0
    assert bgem3_sparse_score(q, d) == pytest.approx(expected)
    assert bgem3_sparse_score({}, d) == pytest.approx(0.0)


def test_colbert_score_maxsim_normalised():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    score = bgem3_colbert_score(q, d)
    # Каждый токен запроса находит идеальный матч в документе → maxsim=1 для каждого, среднее = 1.
    assert score == pytest.approx(1.0, abs=1e-5)


def test_bge_m3_retriever_reranks_pool_by_fused_score():
    pool = [
        RetrievedDoc(text="doc-A", score=0.1, rank=1, chunk_id="A"),
        RetrievedDoc(text="doc-B", score=0.2, rank=2, chunk_id="B"),
        RetrievedDoc(text="doc-C", score=0.3, rank=3, chunk_id="C"),
    ]
    base = _PoolRetriever(pool)
    q_pack = {
        "dense_vecs": np.array([[1.0, 0.0]]),
        "lexical_weights": [{"hot": 1.0}],
        "colbert_vecs": [np.array([[1.0, 0.0]])],
    }
    d_pack = {
        "dense_vecs": np.array([
            [0.0, 1.0],     # A: dense ortho
            [1.0, 0.0],     # B: dense match
            [0.7, 0.0],     # C: dense partial
        ]),
        "lexical_weights": [
            {},                          # A
            {"hot": 1.0},                # B
            {"hot": 0.5},                # C
        ],
        "colbert_vecs": [
            np.array([[0.0, 1.0]]),       # A
            np.array([[1.0, 0.0]]),       # B
            np.array([[1.0, 0.0]]),       # C
        ],
    }
    enc = _CannedEncoder(q_pack, d_pack)
    r = BgeM3MultiVectorRetriever(
        base, enc,
        candidate_top_k=3,
        w_dense=0.4, w_sparse=0.2, w_colbert=0.4,
    )
    res = r.search(RetrievalRequest(query="hot", top_k=3))

    # B должен оказаться первым — dense=1, sparse=1, colbert=1.
    assert [d.chunk_id for d in res.docs][0] == "B"
    # Сохраняется метаданные с per-component скорами.
    md = res.docs[0].metadata
    assert "bge_m3_dense_score" in md
    assert "bge_m3_sparse_score" in md
    assert "bge_m3_colbert_score" in md
    assert md["bge_m3_weights"] == {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}


def test_bge_m3_zero_weights_skip_components():
    pool = [RetrievedDoc(text="d", score=0.0, rank=1, chunk_id="X")]
    base = _PoolRetriever(pool)

    class _BoomEncoder:
        def encode_queries(self, queries):
            return {
                "dense_vecs": np.array([[1.0]]),
                "lexical_weights": [{}],
                "colbert_vecs": [None],
            }

        def encode_documents(self, docs):
            return {
                "dense_vecs": np.array([[1.0]]),
                "lexical_weights": [{}],
                "colbert_vecs": [None],
            }

    r = BgeM3MultiVectorRetriever(
        base, _BoomEncoder(),
        candidate_top_k=1, w_dense=1.0, w_sparse=0.0, w_colbert=0.0,
    )
    res = r.search(RetrievalRequest(query="q", top_k=1))
    md = res.docs[0].metadata
    assert md["bge_m3_sparse_score"] == 0.0
    assert md["bge_m3_colbert_score"] == 0.0
    assert md["bge_m3_dense_score"] == pytest.approx(1.0)


def test_bge_m3_empty_pool_returns_empty():
    base = _PoolRetriever([])
    enc = _CannedEncoder({"dense_vecs": np.array([[1.0]])}, {"dense_vecs": np.array([])})
    r = BgeM3MultiVectorRetriever(base, enc, candidate_top_k=10)
    res = r.search(RetrievalRequest(query="q", top_k=5))
    assert res.docs == []
    assert res.debug.get("reason") == "empty_pool"

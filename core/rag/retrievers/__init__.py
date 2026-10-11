"""Пакет ретриверов.

Конкретные реализации:

- :mod:`core.rag.retrievers.dense_chroma_retriever` — dense через ChromaDB;
- :mod:`core.rag.retrievers.sparse_bm25_retriever`  — BM25 sidecar (F2);
- :mod:`core.rag.retrievers.hybrid_rrf_retriever`   — RRF-fusion (F2);
- :mod:`core.rag.retrievers.hybrid_linear_weighted_retriever` — linear/weighted (F2);
- :mod:`core.rag.retrievers.bge_m3_multi_vector_retriever` — BGE-M3 dense+sparse+ColBERT (F4);
- :mod:`core.rag.retrievers.hyde_query_expansion_retriever` — HyDE (F3);
- :mod:`core.rag.retrievers.multi_query_rewrite_retriever`  — MQR (F3).

Базовые типы и протокол: :mod:`core.rag.retrievers.retriever_base_types`.
"""
from .bge_m3_multi_vector_retriever import (
    BgeM3MultiVectorRetriever,
    FlagEmbeddingMultiVectorEncoder,
    MultiVectorEncoder,
    bgem3_colbert_score,
    bgem3_dense_score,
    bgem3_sparse_score,
)
from .dense_chroma_retriever import DenseChromaRetriever
from .hybrid_linear_weighted_retriever import HybridLinearWeightedRetriever
from .hybrid_rrf_retriever import HybridRrfRetriever
from .hyde_query_expansion_retriever import HydeQueryExpansionRetriever
from .multi_query_rewrite_retriever import MultiQueryRewriteRetriever
from .reranker_wrapping_retriever import RerankerWrappingRetriever
from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)
from .sparse_bm25_retriever import SparseBm25Retriever, build_bm25_index_from_jsonl

__all__ = [
    "Retriever",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedDoc",
    "DenseChromaRetriever",
    "SparseBm25Retriever",
    "build_bm25_index_from_jsonl",
    "HybridRrfRetriever",
    "HybridLinearWeightedRetriever",
    "HydeQueryExpansionRetriever",
    "MultiQueryRewriteRetriever",
    "RerankerWrappingRetriever",
    "BgeM3MultiVectorRetriever",
    "FlagEmbeddingMultiVectorEncoder",
    "MultiVectorEncoder",
    "bgem3_dense_score",
    "bgem3_sparse_score",
    "bgem3_colbert_score",
]

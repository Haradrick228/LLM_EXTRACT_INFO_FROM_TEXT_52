"""Reranker-обёртка над любым :class:`Retriever`.

Pipeline: ``base_retriever`` достаёт ``candidate_top_k`` кандидатов,
``reranker`` пересортирует и обрезает до ``request.top_k``. Идеально для
F7: применить разные cross-encoder реранкеры (BGE-v2-m3, Jina-v3,
Qwen3-Reranker, ColBERT) поверх лучшего retriever-конфига из F2-F6.

При ``reranker is None`` или ``reranker.rerank`` бросает — fallback на
оригинальный порядок ``base_retriever`` (с обрезкой до ``top_k``).
"""
from __future__ import annotations

import time
from typing import List, Optional

from ..rerankers.reranker_base_types import RerankRequest, Reranker
from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)


class RerankerWrappingRetriever:
    """``base_retriever`` → top-N кандидатов → reranker → top-K итог."""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Optional[Reranker],
        *,
        candidate_top_k: int = 50,
        name: Optional[str] = None,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_top_k = int(candidate_top_k)
        rname = getattr(reranker, "name", "none")
        self.name = name or f"rerank({rname})::{getattr(base_retriever, 'name', 'base')}"

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        pool_req = RetrievalRequest(
            query=request.query,
            top_k=max(self.candidate_top_k, request.top_k),
            extra=request.extra,
        )
        pool = self.base_retriever.search(pool_req)
        if not pool.docs or self.reranker is None:
            out_docs = pool.docs[: request.top_k]
            return RetrievalResult(
                docs=out_docs,
                timings_ms={
                    **pool.timings_ms,
                    "rerank_total_ms": int((time.monotonic() - t0) * 1000),
                    "rerank_kind": 0,
                },
                debug={
                    "rerank_skipped": True,
                    "pool_size": len(pool.docs),
                    **pool.debug,
                },
            )

        rr = self.reranker.rerank(
            RerankRequest(query=request.query, docs=pool.docs, top_k=request.top_k)
        )
        reranked: List[RetrievedDoc] = list(rr.docs)
        elapsed = int((time.monotonic() - t0) * 1000)
        return RetrievalResult(
            docs=reranked,
            timings_ms={
                **pool.timings_ms,
                "rerank_only_ms": int(rr.timings_ms or 0),
                "rerank_total_ms": elapsed,
            },
            debug={
                "rerank_skipped": False,
                "rerank_name": getattr(self.reranker, "name", "unknown"),
                "candidate_top_k": self.candidate_top_k,
                "pool_size": len(pool.docs),
                **pool.debug,
            },
        )


__all__ = ["RerankerWrappingRetriever"]

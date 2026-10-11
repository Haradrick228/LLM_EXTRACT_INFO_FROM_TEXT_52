"""Reciprocal Rank Fusion (RRF) ретривер.

Объединяет результаты ``N`` любых ретриверов в одну выдачу. RRF — это
параметр-свободная функция, не требующая нормализации скоров:

::

    RRF(d) = sum_i 1 / (k + rank_i(d))

Документ, отсутствующий в выдаче ретривера ``i``, в сумму вклада не вносит.

Параметр ``k`` (rank-bias) по умолчанию ``60``, как в оригинальной статье
Cormack et al. (2009). Прокидывается аргументом, чтобы можно было свипать
``{40, 60, 80, 100}``.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)


def _doc_key(doc: RetrievedDoc) -> str:
    """Стабильный ключ для дедупа: ``chunk_id`` → ``source`` → префикс текста."""
    if doc.chunk_id:
        return f"cid::{doc.chunk_id}"
    if doc.source:
        return f"src::{doc.source}::{(doc.text or '')[:64]}"
    return f"txt::{(doc.text or '')[:128]}"


class HybridRrfRetriever:
    """Reciprocal Rank Fusion поверх нескольких :class:`Retriever`'ов."""

    def __init__(
        self,
        retrievers: List[Retriever],
        *,
        k: int = 60,
        per_retriever_top_k: Optional[int] = None,
        name: str = "hybrid_rrf",
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRrfRetriever requires at least one underlying retriever")
        self.retrievers = retrievers
        self.k = int(k)
        self.per_retriever_top_k = per_retriever_top_k
        self.name = name

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t_start = time.monotonic()
        per_retriever_top = self.per_retriever_top_k or max(request.top_k, 40)

        accum_score: Dict[str, float] = {}
        # Документ с минимальным rank — сохраняем (он будет в основе результата).
        best_doc: Dict[str, RetrievedDoc] = {}
        # source-trace: какой ретривер каким ранком вернул этот документ
        source_ranks: Dict[str, Dict[str, int]] = {}
        timings: Dict[str, int] = {}

        per_retriever_results: List[Tuple[str, RetrievalResult]] = []

        for retriever in self.retrievers:
            sub_req = RetrievalRequest(
                query=request.query, top_k=per_retriever_top, extra=request.extra,
            )
            r = retriever.search(sub_req)
            per_retriever_results.append((retriever.name, r))
            for tk, tv in r.timings_ms.items():
                timings[f"{retriever.name}.{tk}"] = tv
            for d in r.docs:
                key = _doc_key(d)
                rrf_inc = 1.0 / (self.k + d.rank)
                accum_score[key] = accum_score.get(key, 0.0) + rrf_inc
                if key not in best_doc or d.rank < best_doc[key].rank:
                    best_doc[key] = d
                source_ranks.setdefault(key, {})[retriever.name] = d.rank

        # Сортируем по убыванию RRF-скора, берём top_k.
        ordered_keys = sorted(accum_score.keys(), key=lambda k: accum_score[k], reverse=True)
        out: List[RetrievedDoc] = []
        for new_rank, key in enumerate(ordered_keys[: request.top_k], start=1):
            base = best_doc[key]
            out.append(
                RetrievedDoc(
                    text=base.text,
                    score=float(accum_score[key]),
                    rank=new_rank,
                    chunk_id=base.chunk_id,
                    source=base.source,
                    metadata={
                        **base.metadata,
                        "rrf_score": accum_score[key],
                        "rrf_source_ranks": source_ranks[key],
                        "rrf_k": self.k,
                    },
                )
            )

        elapsed = int((time.monotonic() - t_start) * 1000)
        return RetrievalResult(
            docs=out,
            timings_ms={**timings, "rrf_total_ms": elapsed},
            debug={
                "rrf_k": self.k,
                "per_retriever_top_k": per_retriever_top,
                "n_retrievers": len(self.retrievers),
                "n_unique_candidates": len(accum_score),
            },
        )


__all__ = ["HybridRrfRetriever"]

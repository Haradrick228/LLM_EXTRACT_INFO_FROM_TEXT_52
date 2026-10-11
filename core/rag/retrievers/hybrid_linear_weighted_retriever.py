"""Linear-weighted fusion.

В отличие от RRF, нормализует скоры каждого ретривера в ``[0, 1]`` (min-max)
и считает взвешенную сумму. Веса задаются параметром ``weights`` — `dict`
``{retriever.name -> w}``; отсутствующие имена считаются ``1.0``.

Используется как контрольный сэмпл в свипе F2, чтобы убедиться, что RRF
действительно лучше при текущих данных.
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
    if doc.chunk_id:
        return f"cid::{doc.chunk_id}"
    if doc.source:
        return f"src::{doc.source}::{(doc.text or '')[:64]}"
    return f"txt::{(doc.text or '')[:128]}"


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return values
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridLinearWeightedRetriever:
    """Linear-weighted min-max fusion."""

    def __init__(
        self,
        retrievers: List[Retriever],
        *,
        weights: Optional[Dict[str, float]] = None,
        per_retriever_top_k: Optional[int] = None,
        name: str = "hybrid_linear_weighted",
    ) -> None:
        if not retrievers:
            raise ValueError("HybridLinearWeightedRetriever requires at least one retriever")
        self.retrievers = retrievers
        self.weights = dict(weights or {})
        self.per_retriever_top_k = per_retriever_top_k
        self.name = name

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t_start = time.monotonic()
        per_retriever_top = self.per_retriever_top_k or max(request.top_k, 40)

        accum_score: Dict[str, float] = {}
        best_doc: Dict[str, RetrievedDoc] = {}
        source_ranks: Dict[str, Dict[str, int]] = {}
        timings: Dict[str, int] = {}

        for retriever in self.retrievers:
            sub_req = RetrievalRequest(query=request.query, top_k=per_retriever_top, extra=request.extra)
            r = retriever.search(sub_req)
            for tk, tv in r.timings_ms.items():
                timings[f"{retriever.name}.{tk}"] = tv
            scores_norm = _minmax([d.score for d in r.docs])
            w = float(self.weights.get(retriever.name, 1.0))
            for d, s_norm in zip(r.docs, scores_norm):
                key = _doc_key(d)
                accum_score[key] = accum_score.get(key, 0.0) + w * s_norm
                if key not in best_doc or d.rank < best_doc[key].rank:
                    best_doc[key] = d
                source_ranks.setdefault(key, {})[retriever.name] = d.rank

        ordered = sorted(accum_score.keys(), key=lambda k: accum_score[k], reverse=True)
        out: List[RetrievedDoc] = []
        for new_rank, key in enumerate(ordered[: request.top_k], start=1):
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
                        "linear_weighted_score": accum_score[key],
                        "linear_weighted_source_ranks": source_ranks[key],
                        "linear_weighted_weights": dict(self.weights),
                    },
                )
            )

        elapsed = int((time.monotonic() - t_start) * 1000)
        return RetrievalResult(
            docs=out,
            timings_ms={**timings, "linear_total_ms": elapsed},
            debug={
                "weights": dict(self.weights),
                "n_retrievers": len(self.retrievers),
                "n_unique_candidates": len(accum_score),
                "per_retriever_top_k": per_retriever_top,
            },
        )


__all__ = ["HybridLinearWeightedRetriever"]

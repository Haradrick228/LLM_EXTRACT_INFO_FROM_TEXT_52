"""Типы и протокол для реранкеров.

Реранкер берёт top-fetch_k кандидатов от ретривера и переставляет их
порядок (или отдаёт top-k подмножество). Cross-encoder реализация —
``cross_encoder_reranker``. Late-interaction ColBERT — ``colbert_late_interaction_reranker``
(F4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

from ..retrievers.retriever_base_types import RetrievedDoc


@dataclass(frozen=True)
class RerankRequest:
    query: str
    docs: List[RetrievedDoc]
    top_k: int


@dataclass(frozen=True)
class RerankResult:
    docs: List[RetrievedDoc]
    timings_ms: int = 0


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, request: RerankRequest) -> RerankResult: ...


__all__ = ["RerankRequest", "RerankResult", "Reranker"]

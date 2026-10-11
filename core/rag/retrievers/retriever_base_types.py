"""Базовые типы и протокол для всех ретриверов.

Любой ретривер реализует ``Retriever.search(query, top_k) -> RetrievalResult``.
Это позволяет ``hybrid_rrf_retriever`` объединять разнотипные ретриверы
(dense Chroma, sparse BM25, ColBERT late-interaction, HyDE-обёртка) единым
способом, а ``eval.retrieval.run_retrieval_eval`` — переключаться флагом
``--retriever``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedDoc:
    """Один результат ретрива.

    ``text``       — текст чанка, как он лежит в индексе.
    ``score``      — скор ретривера (cosine для dense, BM25 score для sparse,
                     RRF score для гибрида). Higher = better.
    ``rank``       — 1-based позиция в выдаче.
    ``chunk_id``   — стабильный id (для Chroma — ``metadata.chunk_document_id`` либо doc id).
    ``source``     — путь/имя исходного документа из ``metadata.source`` (если есть).
    ``metadata``   — произвольные ключи (нужно для трассировки).
    """

    text: str
    score: float
    rank: int
    chunk_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 40
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Выдача ретривера + замеренные тайминги."""

    docs: List[RetrievedDoc]
    timings_ms: Dict[str, int] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    """Минимальный контракт ретривера. Реализации — в этом же пакете."""

    name: str

    def search(self, request: RetrievalRequest) -> RetrievalResult: ...


__all__ = [
    "RetrievedDoc",
    "RetrievalRequest",
    "RetrievalResult",
    "Retriever",
]

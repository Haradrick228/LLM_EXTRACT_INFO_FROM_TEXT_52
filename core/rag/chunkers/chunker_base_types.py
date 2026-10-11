"""Базовые типы для чанкеров (F5).

Реализации в фазе F5:

- ``fixed_size_chunker``       — 650/250, текущее поведение;
- ``parent_child_chunker``     — child 256/64 для retrieval, parent 1024/128 для контекста;
- ``semantic_boundary_chunker`` — границы по embedding-дельте.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    parent_id: Optional[str] = None
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkingResult:
    chunks: List[Chunk]
    parents: List[Chunk] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    name: str

    def split(self, document: SourceDocument) -> ChunkingResult: ...


__all__ = ["SourceDocument", "Chunk", "ChunkingResult", "Chunker"]

"""Пакет чанкеров.

Реализации (F5):

- :mod:`core.rag.chunkers.fixed_size_chunker` — baseline 650/250 (текущий).
- :mod:`core.rag.chunkers.parent_child_chunker` — большие parent + мелкие child.
- :mod:`core.rag.chunkers.semantic_boundary_chunker` — границы по similarity.

Все классы удовлетворяют :class:`Chunker` protocol
(см. :mod:`core.rag.chunkers.chunker_base_types`).
"""
from .chunker_base_types import Chunk, Chunker, ChunkingResult, SourceDocument
from .fixed_size_chunker import FixedSizeChunker
from .parent_child_chunker import ParentChildChunker
from .semantic_boundary_chunker import SemanticBoundaryChunker

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingResult",
    "SourceDocument",
    "FixedSizeChunker",
    "ParentChildChunker",
    "SemanticBoundaryChunker",
]

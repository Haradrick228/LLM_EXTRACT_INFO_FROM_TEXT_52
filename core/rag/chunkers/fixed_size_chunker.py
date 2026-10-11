"""Fixed-size chunker (baseline 650/250, word-based).

Аналог текущего поведения (см. ``DATA/runs/corpus_full_export_*``):

- Разбивает входной текст на чанки по словам;
- ``chunk_size`` слов с ``chunk_overlap`` пересечения с соседним чанком;
- ``chunk_id = f"{doc_id}::c{order:04d}"`` — стабилен между прогонами.

Работает по **словам**, потому что наши тексты — русскоязычные, и
символьное окно неустойчиво на длине слов. При желании можно подать
кастомный токенизатор через ``tokenize`` (по умолчанию ``str.split``).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .chunker_base_types import Chunk, ChunkingResult, SourceDocument


def _default_tokenize(text: str) -> List[str]:
    return text.split()


def _default_detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


class FixedSizeChunker:
    """Baseline-чанкер с фиксированным окном и overlap."""

    def __init__(
        self,
        *,
        chunk_size: int = 650,
        chunk_overlap: int = 250,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        detokenize: Optional[Callable[[List[str]], str]] = None,
        name: str = "fixed_size",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.tokenize = tokenize or _default_tokenize
        self.detokenize = detokenize or _default_detokenize
        self.name = name

    def split(self, document: SourceDocument) -> ChunkingResult:
        tokens = self.tokenize(document.text or "")
        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        i = 0
        order = 0
        n = len(tokens)
        while i < n:
            window = tokens[i : i + self.chunk_size]
            if not window:
                break
            text = self.detokenize(window).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::c{order:04d}",
                        text=text,
                        parent_id=None,
                        order=order,
                        metadata={
                            **document.metadata,
                            "chunker": self.name,
                            "chunk_size_tokens": len(window),
                            "chunk_overlap_tokens": self.chunk_overlap,
                            "source_doc_id": document.doc_id,
                            "char_length": len(text),
                        },
                    )
                )
                order += 1
            # последний chunk покрыл конец документа — стоп.
            if i + self.chunk_size >= n:
                break
            i += step
        return ChunkingResult(
            chunks=chunks,
            parents=[],
            debug={
                "chunker": self.name,
                "n_tokens": n,
                "n_chunks": len(chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )


__all__ = ["FixedSizeChunker"]

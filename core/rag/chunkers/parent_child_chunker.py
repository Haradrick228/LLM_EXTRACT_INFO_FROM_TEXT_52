"""Parent-Child chunker: индексируем «детей», возвращаем «родителей» для контекста.

Идея (LangChain ParentDocumentRetriever / LlamaIndex AutoMergingRetriever):

- Большие ``parent``-чанки сохраняют контекст и хорошо подходят для **ответа**
  LLM;
- Мелкие ``child``-чанки лучше **матчатся** по запросу при retrieval
  (повышают recall@k);
- При индексации в Chroma хранятся **child**-эмбеддинги; при ответе
  доставляем родителей этих детей.

Реализация:

- два уровня ``FixedSizeChunker``-логики: сначала разбиваем документ на
  родителей (``parent_size`` / ``parent_overlap``), затем каждого родителя —
  на детей (``child_size`` / ``child_overlap``);
- ``Chunk.parent_id`` у ребёнка хранит ``chunk_id`` родителя;
- ``ChunkingResult.parents`` содержит сами родительские чанки.

Возвращаем чанки в порядке: сначала все child-чанки (в порядке появления),
затем доступ к родителям через ``ChunkingResult.parents`` (та же самая
ссылка, что и ``Chunk.parent_id`` детей).
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .chunker_base_types import Chunk, ChunkingResult, SourceDocument
from .fixed_size_chunker import (
    _default_detokenize,
    _default_tokenize,
)


class ParentChildChunker:
    """Двухуровневый чанкер (parent + child)."""

    def __init__(
        self,
        *,
        parent_size: int = 1024,
        parent_overlap: int = 128,
        child_size: int = 256,
        child_overlap: int = 64,
        tokenize: Optional[Callable[[str], List[str]]] = None,
        detokenize: Optional[Callable[[List[str]], str]] = None,
        name: str = "parent_child",
    ) -> None:
        if parent_size <= 0 or child_size <= 0:
            raise ValueError("parent_size and child_size must be > 0")
        if parent_overlap < 0 or parent_overlap >= parent_size:
            raise ValueError("parent_overlap must be in [0, parent_size)")
        if child_overlap < 0 or child_overlap >= child_size:
            raise ValueError("child_overlap must be in [0, child_size)")
        if child_size > parent_size:
            raise ValueError("child_size must be <= parent_size")
        self.parent_size = int(parent_size)
        self.parent_overlap = int(parent_overlap)
        self.child_size = int(child_size)
        self.child_overlap = int(child_overlap)
        self.tokenize = tokenize or _default_tokenize
        self.detokenize = detokenize or _default_detokenize
        self.name = name

    def _split_window(
        self, tokens: List[str], size: int, overlap: int
    ) -> List[List[str]]:
        if not tokens:
            return []
        step = max(1, size - overlap)
        out: List[List[str]] = []
        i = 0
        n = len(tokens)
        while i < n:
            window = tokens[i : i + size]
            if not window:
                break
            out.append(window)
            if i + size >= n:
                break
            i += step
        return out

    def split(self, document: SourceDocument) -> ChunkingResult:
        tokens = self.tokenize(document.text or "")
        parent_windows = self._split_window(tokens, self.parent_size, self.parent_overlap)

        parents: List[Chunk] = []
        children: List[Chunk] = []
        for p_order, p_window in enumerate(parent_windows):
            p_text = self.detokenize(p_window).strip()
            if not p_text:
                continue
            parent_id = f"{document.doc_id}::p{p_order:04d}"
            parents.append(
                Chunk(
                    chunk_id=parent_id,
                    text=p_text,
                    parent_id=None,
                    order=p_order,
                    metadata={
                        **document.metadata,
                        "chunker": self.name,
                        "role": "parent",
                        "chunk_size_tokens": len(p_window),
                        "source_doc_id": document.doc_id,
                        "char_length": len(p_text),
                    },
                )
            )

            child_windows = self._split_window(p_window, self.child_size, self.child_overlap)
            for c_order, c_window in enumerate(child_windows):
                c_text = self.detokenize(c_window).strip()
                if not c_text:
                    continue
                child_id = f"{parent_id}::c{c_order:04d}"
                children.append(
                    Chunk(
                        chunk_id=child_id,
                        text=c_text,
                        parent_id=parent_id,
                        order=c_order,
                        metadata={
                            **document.metadata,
                            "chunker": self.name,
                            "role": "child",
                            "chunk_size_tokens": len(c_window),
                            "child_overlap_tokens": self.child_overlap,
                            "source_doc_id": document.doc_id,
                            "parent_chunk_id": parent_id,
                            "parent_order": p_order,
                            "char_length": len(c_text),
                        },
                    )
                )

        return ChunkingResult(
            chunks=children,
            parents=parents,
            debug={
                "chunker": self.name,
                "n_tokens": len(tokens),
                "n_parents": len(parents),
                "n_children": len(children),
                "parent_size": self.parent_size,
                "parent_overlap": self.parent_overlap,
                "child_size": self.child_size,
                "child_overlap": self.child_overlap,
            },
        )


__all__ = ["ParentChildChunker"]

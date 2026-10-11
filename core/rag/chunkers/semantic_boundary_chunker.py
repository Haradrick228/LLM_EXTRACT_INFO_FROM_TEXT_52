"""Semantic-boundary chunker: режем по «семантическим швам».

Подход (см. Greg Kamradt's «5 Levels of Text Splitting», 2024):

1. Бьём текст на предложения (regex).
2. Эмбеддим каждое предложение (через :class:`SentenceEmbeddingFn`-протокол).
3. Считаем cosine similarity между соседними предложениями.
4. Граница ставится там, где similarity падает ниже ``threshold`` *или* ниже
   ``percentile``-перцентиля распределения (тот, что меньше — тот сильнее
   режет).
5. Аккумулированные предложения сшиваются в чанки; если суммарный размер
   чанка превышает ``max_chunk_tokens`` — принудительно режем (защита от
   гигантских блоков).

Эмбеддер инъецируется (для тестов используем фейковый, в production — обёртку
вокруг ``SentenceTransformer``). Минимально:

::

    def embed(sentences: List[str]) -> List[List[float]]: ...

Опциональный аргумент ``min_sentences_per_chunk`` гарантирует, что чанк
содержит не менее ``N`` предложений.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from .chunker_base_types import Chunk, ChunkingResult, SourceDocument

_SENTENCE_END = re.compile(r"(?<=[\.\?!…])\s+(?=[A-ZА-ЯЁ])", flags=re.UNICODE)


def split_sentences(text: str) -> List[str]:
    """Дешёвый regex-сплиттер на предложения для RU/EN текста.

    Не идеальный (см. инициалы, аббревиатуры), но дешёвый и устойчивый.
    Для production-grade можно подменить через DI.
    """
    if not text:
        return []
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


SentenceEmbeddingFn = Callable[[List[str]], List[List[float]]]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return s / (na * nb)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


class SemanticBoundaryChunker:
    """Семантический чанкер с порогом + перцентилем."""

    def __init__(
        self,
        embedder: SentenceEmbeddingFn,
        *,
        sentence_splitter: Callable[[str], List[str]] = split_sentences,
        similarity_threshold: float = 0.55,
        breakpoint_percentile: float = 0.2,
        max_chunk_tokens: int = 768,
        min_sentences_per_chunk: int = 1,
        name: str = "semantic_boundary",
    ) -> None:
        if not 0.0 <= breakpoint_percentile <= 1.0:
            raise ValueError("breakpoint_percentile must be in [0, 1]")
        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be > 0")
        self.embedder = embedder
        self.sentence_splitter = sentence_splitter
        self.similarity_threshold = float(similarity_threshold)
        self.breakpoint_percentile = float(breakpoint_percentile)
        self.max_chunk_tokens = int(max_chunk_tokens)
        self.min_sentences_per_chunk = max(1, int(min_sentences_per_chunk))
        self.name = name

    def _breakpoints(self, similarities: List[float]) -> set[int]:
        """Между ``i`` и ``i+1`` ставим breakpoint, если sim падает."""
        if not similarities:
            return set()
        perc = _percentile(similarities, self.breakpoint_percentile)
        cutoff = min(self.similarity_threshold, perc)
        return {i for i, s in enumerate(similarities) if s < cutoff}

    def split(self, document: SourceDocument) -> ChunkingResult:
        sentences = self.sentence_splitter(document.text or "")
        if not sentences:
            return ChunkingResult(chunks=[], parents=[], debug={"chunker": self.name, "n_sentences": 0})

        if len(sentences) == 1:
            text = sentences[0]
            return ChunkingResult(
                chunks=[
                    Chunk(
                        chunk_id=f"{document.doc_id}::s0000",
                        text=text,
                        order=0,
                        metadata={
                            **document.metadata,
                            "chunker": self.name,
                            "source_doc_id": document.doc_id,
                            "n_sentences": 1,
                            "char_length": len(text),
                        },
                    )
                ],
                parents=[],
                debug={"chunker": self.name, "n_sentences": 1, "n_chunks": 1},
            )

        embeddings = self.embedder(sentences)
        if len(embeddings) != len(sentences):
            raise ValueError(
                f"embedder returned {len(embeddings)} vectors for {len(sentences)} sentences"
            )
        similarities = [
            _cosine(embeddings[i], embeddings[i + 1]) for i in range(len(sentences) - 1)
        ]
        breakpoints = self._breakpoints(similarities)

        chunks: List[Chunk] = []
        cur: List[str] = []
        cur_token_count = 0
        order = 0

        def _flush() -> None:
            nonlocal cur, cur_token_count, order
            if not cur:
                return
            text = " ".join(cur).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::s{order:04d}",
                        text=text,
                        order=order,
                        metadata={
                            **document.metadata,
                            "chunker": self.name,
                            "source_doc_id": document.doc_id,
                            "n_sentences": len(cur),
                            "char_length": len(text),
                        },
                    )
                )
                order += 1
            cur = []
            cur_token_count = 0

        for i, sentence in enumerate(sentences):
            cur.append(sentence)
            cur_token_count += len(sentence.split())
            # граница после i-го предложения = breakpoint at index i (если есть)
            at_breakpoint = i in breakpoints
            at_max = cur_token_count >= self.max_chunk_tokens
            if (at_breakpoint and len(cur) >= self.min_sentences_per_chunk) or at_max:
                _flush()
        _flush()

        return ChunkingResult(
            chunks=chunks,
            parents=[],
            debug={
                "chunker": self.name,
                "n_sentences": len(sentences),
                "n_chunks": len(chunks),
                "n_breakpoints": len(breakpoints),
                "similarity_threshold": self.similarity_threshold,
                "breakpoint_percentile": self.breakpoint_percentile,
                "max_chunk_tokens": self.max_chunk_tokens,
            },
        )


__all__ = ["SemanticBoundaryChunker", "split_sentences", "SentenceEmbeddingFn"]

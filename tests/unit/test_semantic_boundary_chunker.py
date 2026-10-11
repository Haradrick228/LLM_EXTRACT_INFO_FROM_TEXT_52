"""SemanticBoundaryChunker: разрезает по «семантическим швам»."""
from __future__ import annotations

from typing import List

import pytest

from core.rag.chunkers.chunker_base_types import SourceDocument
from core.rag.chunkers.semantic_boundary_chunker import (
    SemanticBoundaryChunker,
    _cosine,
    _percentile,
    split_sentences,
)


def _doc(text: str) -> SourceDocument:
    return SourceDocument(doc_id="D", text=text)


def test_split_sentences_splits_ru_en():
    text = (
        "Это первое предложение. Это второе! А вот третье? "
        "And here is the English sentence. The next one starts after a dot."
    )
    out = split_sentences(text)
    assert len(out) == 5
    assert out[0].startswith("Это первое")


def test_cosine_helpers():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine([], [1.0]) == 0.0


def test_percentile_helper():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert _percentile(vals, 0.0) == pytest.approx(0.1)
    assert _percentile(vals, 1.0) == pytest.approx(0.5)
    assert _percentile(vals, 0.5) == pytest.approx(0.3)


def test_breakpoint_splits_when_similarity_drops():
    """Эмбеддер делает 3 «темы»: 0..1 — A, 2..3 — B, 4..5 — C."""
    sentences_groups = [
        ["First about cats.", "Cats are nice animals."],
        ["Cars are vehicles.", "Vehicles have engines."],
        ["Cooking is fun.", "Use a knife carefully."],
    ]
    text = " ".join(s for grp in sentences_groups for s in grp)

    def fake_embedder(sentences: List[str]) -> List[List[float]]:
        # 3-D one-hot per group
        out: List[List[float]] = []
        idx = 0
        for group_id, grp in enumerate(sentences_groups):
            vec = [0.0, 0.0, 0.0]
            vec[group_id] = 1.0
            for _ in grp:
                out.append(vec.copy())
                idx += 1
        return out[: len(sentences)]

    chunker = SemanticBoundaryChunker(
        fake_embedder,
        similarity_threshold=0.5,
        breakpoint_percentile=1.0,  # cutoff = threshold
        max_chunk_tokens=10_000,
    )
    out = chunker.split(_doc(text))

    # 3 groups → 3 chunks
    assert len(out.chunks) == 3
    for i, c in enumerate(out.chunks):
        assert c.metadata["n_sentences"] == 2
        assert c.metadata["chunker"] == "semantic_boundary"
        assert c.chunk_id == f"D::s{i:04d}"


def test_max_chunk_tokens_forces_flush():
    sentences = [
        "One two three four five.",
        "Six seven eight nine ten.",
        "Eleven twelve thirteen.",
    ]
    text = " ".join(sentences)

    def all_similar_embedder(s: List[str]) -> List[List[float]]:
        return [[1.0, 0.0] for _ in s]

    chunker = SemanticBoundaryChunker(
        all_similar_embedder,
        similarity_threshold=0.5,
        breakpoint_percentile=0.0,  # ничего не выпиливаем по similarity
        max_chunk_tokens=5,
    )
    out = chunker.split(_doc(text))
    # max_chunk_tokens=5 → каждое предложение >= 5 слов → flush после каждого
    assert len(out.chunks) == 3


def test_empty_doc():
    out = SemanticBoundaryChunker(lambda s: [[0.0]]).split(_doc(""))
    assert out.chunks == []


def test_single_sentence_returns_one_chunk():
    out = SemanticBoundaryChunker(lambda s: [[1.0]]).split(_doc("single sentence here"))
    assert len(out.chunks) == 1
    assert out.chunks[0].text == "single sentence here"

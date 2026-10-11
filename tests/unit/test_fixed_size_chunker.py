"""FixedSizeChunker: окно с overlap, стабильные id, контракт ChunkingResult."""
from __future__ import annotations

import pytest

from core.rag.chunkers.chunker_base_types import SourceDocument
from core.rag.chunkers.fixed_size_chunker import FixedSizeChunker


def _doc(text: str, *, doc_id: str = "D") -> SourceDocument:
    return SourceDocument(doc_id=doc_id, text=text, metadata={"src": doc_id})


def test_chunk_size_and_overlap_basic():
    tokens = [f"w{i}" for i in range(10)]
    text = " ".join(tokens)
    out = FixedSizeChunker(chunk_size=4, chunk_overlap=2).split(_doc(text))
    expected_windows = [
        ["w0", "w1", "w2", "w3"],
        ["w2", "w3", "w4", "w5"],
        ["w4", "w5", "w6", "w7"],
        ["w6", "w7", "w8", "w9"],
    ]
    assert [c.text.split() for c in out.chunks] == expected_windows
    assert [c.chunk_id for c in out.chunks] == [
        "D::c0000", "D::c0001", "D::c0002", "D::c0003",
    ]
    assert all(c.parent_id is None for c in out.chunks)
    assert out.parents == []
    assert out.debug["n_tokens"] == 10
    assert out.debug["n_chunks"] == 4


def test_no_overlap_partitions_exactly():
    text = " ".join(f"w{i}" for i in range(6))
    out = FixedSizeChunker(chunk_size=3, chunk_overlap=0).split(_doc(text))
    assert len(out.chunks) == 2
    assert out.chunks[0].text == "w0 w1 w2"
    assert out.chunks[1].text == "w1 w2 w3" or out.chunks[1].text == "w3 w4 w5"
    # actually no overlap → step=3 → ["w0..w2", "w3..w5"]
    assert out.chunks[1].text == "w3 w4 w5"


def test_empty_text_yields_no_chunks():
    out = FixedSizeChunker(chunk_size=10, chunk_overlap=2).split(_doc(""))
    assert out.chunks == []
    assert out.debug["n_tokens"] == 0


def test_short_text_one_chunk():
    out = FixedSizeChunker(chunk_size=100, chunk_overlap=10).split(_doc("hello world"))
    assert len(out.chunks) == 1
    assert out.chunks[0].text == "hello world"
    assert out.chunks[0].metadata["src"] == "D"


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, chunk_overlap=0)

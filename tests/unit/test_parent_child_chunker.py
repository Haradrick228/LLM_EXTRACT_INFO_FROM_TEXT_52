"""ParentChildChunker: parent_id, parents-список, метаданные role."""
from __future__ import annotations

import pytest

from core.rag.chunkers.chunker_base_types import SourceDocument
from core.rag.chunkers.parent_child_chunker import ParentChildChunker


def _doc(text: str) -> SourceDocument:
    return SourceDocument(doc_id="D", text=text, metadata={"src": "D"})


def test_parents_and_children_consistency():
    tokens = [f"w{i}" for i in range(20)]
    text = " ".join(tokens)
    out = ParentChildChunker(
        parent_size=8, parent_overlap=2,
        child_size=4, child_overlap=2,
    ).split(_doc(text))

    # parent 8/2 over 20 tokens → windows at 0,6,12 → 3 parents
    # but last window [12..19] (8 tokens) — full, then i+=6=18, window [18,19] short → stop
    # → parents: w0..w7, w6..w13, w12..w19
    assert len(out.parents) == 3
    assert all(c.parent_id is None for c in out.parents)
    parent_ids = {p.chunk_id for p in out.parents}

    for ch in out.chunks:
        assert ch.parent_id in parent_ids
        assert ch.metadata["role"] == "child"
        assert ch.metadata["parent_chunk_id"] == ch.parent_id
        assert ch.metadata["chunker"] == "parent_child"


def test_invalid_sizes():
    with pytest.raises(ValueError):
        ParentChildChunker(parent_size=10, parent_overlap=0, child_size=20, child_overlap=0)
    with pytest.raises(ValueError):
        ParentChildChunker(parent_size=10, parent_overlap=10, child_size=4, child_overlap=0)
    with pytest.raises(ValueError):
        ParentChildChunker(parent_size=10, parent_overlap=0, child_size=4, child_overlap=4)


def test_empty_doc_returns_empty_lists():
    out = ParentChildChunker().split(_doc(""))
    assert out.chunks == []
    assert out.parents == []


def test_metadata_carries_source_fields():
    text = " ".join(f"w{i}" for i in range(16))
    out = ParentChildChunker(
        parent_size=8, parent_overlap=0, child_size=4, child_overlap=0,
    ).split(_doc(text))
    for p in out.parents:
        assert p.metadata["role"] == "parent"
        assert p.metadata["src"] == "D"
    for c in out.chunks:
        assert c.metadata["role"] == "child"
        assert c.metadata["src"] == "D"

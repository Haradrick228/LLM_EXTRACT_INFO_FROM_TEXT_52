"""SparseBm25Retriever: build_bm25_index_from_jsonl → from_dir → search."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from core.rag.retrievers.retriever_base_types import RetrievalRequest
from core.rag.retrievers.sparse_bm25_retriever import (
    INDEX_FILE_BM25,
    INDEX_FILE_DOCS,
    INDEX_FILE_IDS,
    INDEX_FILE_META,
    SparseBm25Retriever,
    build_bm25_index_from_jsonl,
)


def _write_jsonl(tmp: Path, rows: list[dict]) -> Path:
    p = tmp / "corpus.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def test_build_index_and_query(tmp_path: Path):
    rows = [
        {"id": "c1", "document": "Рефлексия — это способность ИИ-агентов оценивать свои действия.", "metadata": {"source": "doc1"}},
        {"id": "c2", "document": "Капибары очень милые животные.", "metadata": {"source": "doc2"}},
        {"id": "c3", "document": "В RAG-системах используют ретриверы и реранкеры.", "metadata": {"source": "doc3"}},
        {"id": "c4", "document": "Self-reflection помогает агентам улучшаться.", "metadata": {"source": "doc4"}},
    ]
    jpath = _write_jsonl(tmp_path, rows)
    idx_dir = tmp_path / "bm25_idx"
    meta = build_bm25_index_from_jsonl(jsonl_path=jpath, out_dir=idx_dir)
    assert meta["corpus_size"] == 4
    for fname in (INDEX_FILE_BM25, INDEX_FILE_DOCS, INDEX_FILE_IDS, INDEX_FILE_META):
        assert (idx_dir / fname).exists()

    r = SparseBm25Retriever.from_dir(idx_dir)
    res = r.search(RetrievalRequest(query="рефлексия агентов", top_k=2))
    assert res.docs, "expected at least one hit for keyword query"
    top_ids = {d.chunk_id for d in res.docs}
    # c1 и c4 — про рефлексию/агентов.
    assert "c1" in top_ids or "c4" in top_ids
    # c2 (капибары) точно не должна быть на первом месте.
    assert res.docs[0].chunk_id != "c2"
    assert res.timings_ms.get("bm25_query_ms", -1) >= 0
    assert all(d.rank == i + 1 for i, d in enumerate(res.docs))


def test_empty_query_returns_no_docs(tmp_path: Path):
    rows = [{"id": "c1", "document": "hello world", "metadata": {"source": "s"}}]
    jpath = _write_jsonl(tmp_path, rows)
    idx_dir = tmp_path / "bm25_idx"
    build_bm25_index_from_jsonl(jsonl_path=jpath, out_dir=idx_dir)
    r = SparseBm25Retriever.from_dir(idx_dir)
    res = r.search(RetrievalRequest(query="   ", top_k=5))
    assert res.docs == []

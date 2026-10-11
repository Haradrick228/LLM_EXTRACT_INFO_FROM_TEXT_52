import json
import tempfile
from pathlib import Path

from eval.indexing.jsonl_corpus_stream import (
    chroma_safe_metadata,
    count_nonempty_lines,
    iter_jsonl_batches,
    iter_jsonl_chunk_rows,
)


def test_count_nonempty_lines():
    d = Path(tempfile.mkdtemp())
    p = d / "a.jsonl"
    p.write_text("{}\n\n{}\n".replace("{}", json.dumps({"id": "1", "document": "a", "metadata": {}})), encoding="utf-8")
    assert count_nonempty_lines(p) == 2


def test_iter_jsonl_chunk_rows_clean():
    d = Path(tempfile.mkdtemp())
    p = d / "b.jsonl"
    p.write_text(
        json.dumps({"id": "z", "document": "x\x00y", "metadata": {"n": 1}}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_jsonl_chunk_rows(p, apply_clean_text=True))
    assert rows[0][0] == "z"
    assert rows[0][1] == "x y"
    assert rows[0][2] == {"n": 1, "chunk_document_id": "z"}


def test_chroma_safe_metadata_nested():
    m = chroma_safe_metadata({"a": 1, "b": [1, 2]})
    assert m["a"] == 1
    assert isinstance(m["b"], str)
    assert "[1, 2]" in m["b"]


def test_iter_jsonl_batches_respects_max_rows():
    d = Path(tempfile.mkdtemp())
    p = d / "c.jsonl"
    lines = [json.dumps({"id": str(i), "document": str(i), "metadata": {}}) for i in range(10)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = 0
    for ids, docs, _ in iter_jsonl_batches(p, 4, max_rows=9):
        total += len(ids)
    assert total == 9

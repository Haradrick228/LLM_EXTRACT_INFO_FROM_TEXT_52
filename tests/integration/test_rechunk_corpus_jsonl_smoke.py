"""Smoke-тест CLI ``eval.indexing.rechunk_corpus_jsonl``.

Без зависимостей от sentence-transformers: гоняем только ``fixed`` и
``parent_child`` чанкеры. Проверяем формат выходного JSONL и manifest.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _write_in_jsonl(tmp: Path) -> Path:
    p = tmp / "in.jsonl"
    rows = [
        {
            "id": "doc1",
            "document": " ".join(f"w{i}" for i in range(40)),
            "metadata": {"source": "src1"},
        },
        {
            "id": "doc2",
            "document": "short text only.",
            "metadata": {"source": "src2"},
        },
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "eval.indexing.rechunk_corpus_jsonl", *args]
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)


def test_rechunk_fixed_writes_jsonl_and_manifest(tmp_path: Path):
    in_jsonl = _write_in_jsonl(tmp_path)
    out_jsonl = tmp_path / "out_fixed.jsonl"
    res = _run_cli(
        [
            "--in-jsonl", str(in_jsonl),
            "--out-jsonl", str(out_jsonl),
            "--chunker", "fixed",
            "--chunk-size", "10",
            "--chunk-overlap", "2",
        ]
    )
    assert res.returncode == 0, res.stderr
    assert out_jsonl.is_file()
    lines = [l for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l]
    rows = [json.loads(l) for l in lines]
    # doc1: 40 tokens, size=10, overlap=2 → step=8 → windows 0,8,16,24,32 → 5 chunks
    doc1_chunks = [r for r in rows if r["id"].startswith("doc1::")]
    assert len(doc1_chunks) >= 4
    assert all(r["metadata"]["chunker"] == "fixed_size" for r in rows)

    manifest = json.loads((tmp_path / "rechunk_manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunker"] == "fixed_size"
    assert manifest["n_docs"] == 2
    assert manifest["n_chunks"] == len(rows)


def test_rechunk_parent_child_writes_parent_text_in_meta(tmp_path: Path):
    in_jsonl = _write_in_jsonl(tmp_path)
    out_jsonl = tmp_path / "out_pc.jsonl"
    res = _run_cli(
        [
            "--in-jsonl", str(in_jsonl),
            "--out-jsonl", str(out_jsonl),
            "--chunker", "parent_child",
            "--parent-size", "16",
            "--parent-overlap", "0",
            "--child-size", "4",
            "--child-overlap", "0",
        ]
    )
    assert res.returncode == 0, res.stderr
    lines = [l for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l]
    rows = [json.loads(l) for l in lines]
    assert rows
    # дети должны иметь parent_chunk_id и parent_text в metadata
    sample = next(r for r in rows if r["id"].startswith("doc1::"))
    md = sample["metadata"]
    assert md["chunker"] == "parent_child"
    assert md["role"] == "child"
    assert "parent_chunk_id" in md
    assert "parent_text" in md
    assert md["parent_chunk_id"].startswith("doc1::p")

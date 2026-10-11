"""Tests for run_answer_eval_in_run_dir manifest alignment."""
from __future__ import annotations

import json
from pathlib import Path

from eval.diagnostics.run_answer_eval_in_run_dir import _load_retrieval_meta


def test_load_retrieval_meta_roundtrip(tmp_path: Path) -> None:
    meta = {
        "eval_file": str(tmp_path / "custom.jsonl"),
        "fetch_k": 99,
        "embed_model": "test-embed",
        "chroma_collection": "col_x",
        "chroma_host": "10.0.0.1",
        "chroma_port": 9999,
    }
    p = tmp_path / "retrieval.json"
    p.write_text(json.dumps({"recall@10": 0.1, "_meta": meta}, indent=2), encoding="utf-8")
    loaded = _load_retrieval_meta(tmp_path)
    assert loaded == meta

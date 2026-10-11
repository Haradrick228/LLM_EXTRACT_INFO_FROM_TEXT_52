"""Smoke tests for semantic_threshold_probe."""
from __future__ import annotations

import json
from pathlib import Path

from eval.diagnostics.semantic_threshold_probe import run_probe


def test_run_probe_empty_trace(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text("", encoding="utf-8")
    gold = tmp_path / "g.jsonl"
    gold.write_text(
        json.dumps({"question": "q", "references": ["r"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rep = run_probe(
        trace_path=trace,
        gold_path=gold,
        semantic_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_rows=10,
        device=None,
    )
    assert rep["n_sampled_rows"] == 0

"""Tests for eval.diagnostics.classify_rag_errors."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.diagnostics.classify_rag_errors import classify_rag_errors


def _write_trace(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_gold(path: Path, questions: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps({"question": q, "references": ["эталонный текст"]}, ensure_ascii=False) + "\n")


def _write_answer(path: Path, pairs: list[tuple[str, float | None]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for q, sim in pairs:
            f.write(json.dumps({"question": q, "ref_sim": sim}, ensure_ascii=False) + "\n")


def test_type1_retrieval_miss(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    trace = tmp_path / "trace.jsonl"
    _write_gold(gold, ["q1", "q2"])
    _write_trace(
        trace,
        [
            {
                "record": "query_trace",
                "eval_row_index": 0,
                "question_text": "q1",
                "metrics_by_mode": {"substring": {"recall@10": 0.0}},
            },
            {
                "record": "query_trace",
                "eval_row_index": 1,
                "question_text": "q2",
                "metrics_by_mode": {"substring": {"recall@10": 1.0}},
            },
        ],
    )
    ans = tmp_path / "ans.jsonl"
    _write_answer(ans, [("q1", 0.9), ("q2", 0.3)])

    report = classify_rag_errors(
        trace_path=trace,
        gold_path=gold,
        answer_path=ans,
        ref_sim_threshold=0.55,
    )
    assert report["counts"]["type1_retrieval"] == 1
    assert report["counts"]["type2_generation"] == 1
    assert report["counts"].get("type0_ok", 0) == 0


def test_type0_ok_when_high_ref_sim(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    trace = tmp_path / "trace.jsonl"
    _write_gold(gold, ["q1"])
    _write_trace(
        trace,
        [
            {
                "record": "query_trace",
                "eval_row_index": 0,
                "question_text": "q1",
                "metrics_by_mode": {"substring": {"recall@10": 1.0}},
            },
        ],
    )
    ans = tmp_path / "ans.jsonl"
    _write_answer(ans, [("q1", 0.66)])

    report = classify_rag_errors(trace_path=trace, gold_path=gold, answer_path=ans)
    assert report["counts"]["type0_ok"] == 1


def test_retrieval_only_without_answer(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    trace = tmp_path / "trace.jsonl"
    _write_gold(gold, ["q1"])
    _write_trace(
        trace,
        [
            {
                "record": "query_trace",
                "eval_row_index": 0,
                "question_text": "q1",
                "metrics_by_mode": {"substring": {"recall@10": 1.0}},
            },
        ],
    )
    report = classify_rag_errors(trace_path=trace, gold_path=gold, answer_path=None)
    assert report["counts"]["type0_ok"] == 1
    assert report["has_answer_eval"] is False

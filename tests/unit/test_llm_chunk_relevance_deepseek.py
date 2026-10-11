"""Tests for eval.diagnostics.llm_chunk_relevance_deepseek parsing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.diagnostics.llm_chunk_relevance_deepseek import parse_rubric_score, run_llm_scores


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", 2),
        ("Score: 1", 1),
        ("0\nextra", 0),
        ("The answer is 2.", 2),
        ("no digits here", 0),
        ("", 0),
    ],
)
def test_parse_rubric_score(text: str, expected: int) -> None:
    assert parse_rubric_score(text) == expected


def test_run_llm_scores_dry_run(tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps(
            {
                "record": "query_trace",
                "eval_row_index": 0,
                "question_text": "What is X?",
                "stage_01_dense_chroma_retrieval": {
                    "candidates_ordered_by_chroma": [
                        {"text": "chunk a about X"},
                        {"text": "noise"},
                    ]
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    summary = run_llm_scores(
        trace_path=trace,
        out_jsonl=out,
        prompt_version="vtest",
        max_queries=2,
        top_chunks=2,
        dry_run=True,
        model="deepseek-chat",
        sleep_s=0.0,
    )
    assert summary["dry_run"] is True
    assert summary["rows_appended"] >= 1
    assert out.is_file()

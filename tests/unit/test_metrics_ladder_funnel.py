"""Unit tests for eval.diagnostics.metrics_ladder_funnel."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.diagnostics.metrics_ladder_funnel import build_funnel_report


def _write(tmp: Path, name: str, lines: list[str]) -> Path:
    p = tmp / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_funnel_classes_and_conditional(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"question": "q0", "references": ["REF0"]}, ensure_ascii=False)
        + "\n"
        + json.dumps({"question": "q1", "references": ["MISSING"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"id": "c1", "document": "xxx REF0 yyy"}, ensure_ascii=False) + "\n"
        + json.dumps({"id": "c2", "document": "noise only"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def trace_line(idx: int, pre_texts: list[str], post_texts: list[str] | None = None) -> str:
        st1 = {
            "candidates_ordered_by_chroma": [
                {"rank_after_chroma": i + 1, "text": t} for i, t in enumerate(pre_texts)
            ]
        }
        o: dict = {
            "record": "query_trace",
            "eval_row_index": idx,
            "stage_01_dense_chroma_retrieval": st1,
        }
        if post_texts is not None:
            o["stage_02_rerank"] = {
                "candidates_ordered_after_rerank": [
                    {"rank_after_rerank": i + 1, "text": t, "match_ambiguous": False}
                    for i, t in enumerate(post_texts)
                ]
            }
        return json.dumps(o, ensure_ascii=False)

    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        trace_line(0, ["noise", "REF0 here"], ["REF0 here", "noise"])
        + "\n"
        + trace_line(1, ["only noise", "more noise"], None)
        + "\n",
        encoding="utf-8",
    )

    rep = build_funnel_report(
        trace_path=trace,
        gold_path=gold,
        corpus_path=corpus,
        hit_name="substring",
        k_primary=10,
    )
    cc = rep["class_counts"]
    assert cc["D"] == 1
    assert cc["A"] == 1
    assert rep["conditional_given_oracle_pool"]["n"] == 1
    assert rep["rerank"]["n_traces_with_stage2"] == 1
    assert rep["rerank"]["mean_delta_mrr"] is not None


def test_funnel_without_corpus_skips_class_a(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        json.dumps({"question": "q0", "references": ["ZZZNOTINCORPUS"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "record": "query_trace",
                "eval_row_index": 0,
                "stage_01_dense_chroma_retrieval": {
                    "candidates_ordered_by_chroma": [{"rank_after_chroma": 1, "text": "no match here"}]
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rep = build_funnel_report(
        trace_path=trace,
        gold_path=gold,
        corpus_path=None,
        hit_name="substring",
        k_primary=5,
    )
    assert rep["class_counts"]["B"] == 1
    assert rep["class_counts"]["A"] == 0

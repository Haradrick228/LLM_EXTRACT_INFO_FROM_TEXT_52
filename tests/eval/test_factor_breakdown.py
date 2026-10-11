"""Factor breakdown: embedder / reranker marginals vs full cell."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.reporting.decision_analysis import (
    embedder_factor_table_rows,
    pipeline_decomposition_header_row,
    reranker_factor_table_rows,
)

_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = _ROOT / "eval/runs/20260527_161527_b3_full_corpus_rerank/summary.json"
MK = "substring::recall@10"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_PATH.is_file():
        pytest.skip(f"missing {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_best_cell_is_combination_not_single_factor(summary: dict) -> None:
    header = pipeline_decomposition_header_row(summary, MK)
    assert header[0] == "Комбинация (embedder×reranker)"
    assert "full_frida" in header[1]
    assert "6.4%" in header[3]


def test_embedder_ranking_has_six_rows(summary: dict) -> None:
    rows = embedder_factor_table_rows(summary, MK)
    assert len(rows) == 6
    assert rows[0][0] == "full_frida"
    assert rows[0][2] == "6.4%"


def test_reranker_ranking_has_three_rows(summary: dict) -> None:
    rows = reranker_factor_table_rows(summary, MK)
    assert len(rows) == 3
    top_ids = {r[0] for r in rows}
    assert top_ids == {
        "rerank_jina_v3",
        "rerank_bge_v2_m3",
        "baseline_mpnet_rerank",
    }

"""Smoke tests for decision_analysis winners (canonical B3 summary)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.reporting.decision_analysis import (
    PRIMARY_METRIC,
    compute_metric_winners,
    metric_value,
)

_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = _ROOT / "eval/runs/20260527_161527_b3_full_corpus_rerank/summary.json"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_PATH.is_file():
        pytest.skip(f"missing {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_substring_recall_leader_tie(summary: dict) -> None:
    winners = {w.metric_key: w for w in compute_metric_winners(summary)}
    w = winners[PRIMARY_METRIC]
    assert abs(w.value - 0.06435643564356436) < 1e-6
    assert w.scenario == "full_frida"
    assert len(w.tied) >= 2


def test_token_f1_leader_is_bge(summary: dict) -> None:
    winners = {w.metric_key: w for w in compute_metric_winners(summary)}
    w = winners["token_f1::recall@10"]
    assert w.scenario == "full_bge_m3"
    assert w.profile == "rerank_bge_v2_m3"
    assert abs(w.value - 0.0693069306930693) < 1e-6


def test_semantic_recall_leader_is_jina_embed(summary: dict) -> None:
    winners = {w.metric_key: w for w in compute_metric_winners(summary)}
    w = winners["semantic::recall@10"]
    assert w.scenario == "full_jina_embeddings_v3"
    assert w.profile == "rerank_bge_v2_m3"
    assert abs(w.value - 0.7128712871287128) < 1e-5


def test_bge_semantic_mrr_value(summary: dict) -> None:
    cell = summary["results"]["full_bge_m3"]["rerank_bge_v2_m3"]
    mrr = metric_value(cell, "semantic::mrr")
    assert mrr is not None
    assert abs(mrr - 0.6079034002579984) < 1e-6

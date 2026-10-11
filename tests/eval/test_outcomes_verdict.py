"""Outcomes verdict: prod pick and factor leaders."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.reporting.outcomes_verdict import build_outcomes_model

_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = _ROOT / "eval/runs/20260527_161527_b3_full_corpus_rerank/summary.json"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_PATH.is_file():
        pytest.skip(f"missing {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


class _Ans:
    def __init__(self, scenario: str, profile: str, sim: float, r10: float) -> None:
        self.rank = 0
        self.scenario = scenario
        self.profile = profile
        self.avg_ref_sim = sim
        self.retrieval_recall10 = r10
        self.count_ref = 101
        self.ok = True
        self.avg_refs_sim = None


def test_prod_pick_user_bge_overrides_frida_answer_leader(summary: dict) -> None:
    rows = [
        _Ans("full_frida", "rerank_jina_v3", 0.640, 0.0644),
        _Ans("full_frida", "rerank_bge_v2_m3", 0.650, 0.0644),
        _Ans("full_user_bge_m3", "rerank_bge_v2_m3", 0.620, 0.054),
        _Ans("full_bge_m3", "rerank_bge_v2_m3", 0.660, 0.054),
    ]
    model = build_outcomes_model(summary, n_questions=101, n_chunks=61543, fetch_k=40, answer_rows=rows)
    assert model.embedder.factor_id == "full_frida"
    assert model.combo_production.scenario == "full_user_bge_m3"
    assert model.combo_production.profile == "rerank_bge_v2_m3"
    assert model.combo_production.avg_ref_sim == pytest.approx(0.620, abs=1e-6)
    assert model.quality_alternate is not None
    assert model.quality_alternate.scenario == "full_bge_m3"
    assert model.quality_alternate.profile == "rerank_bge_v2_m3"
    assert model.quality_alternate.avg_ref_sim == pytest.approx(0.660, abs=1e-6)

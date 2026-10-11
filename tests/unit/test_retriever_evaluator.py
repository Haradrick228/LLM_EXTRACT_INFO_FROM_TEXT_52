"""evaluate_retriever работает поверх протокола Retriever без сети."""
from __future__ import annotations

import io
import json
from typing import List

import pytest

from core.rag.metrics.substring_match_hit import substring_match_hit
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)
from eval.retrieval.retriever_evaluator import evaluate_retriever


class _CannedRetriever:
    """Возвращает выдачу, заданную мапой question → list[doc]. Имя в трассе сохраняется."""

    def __init__(self, name: str, mapping: dict[str, List[RetrievedDoc]]) -> None:
        self.name = name
        self._mapping = mapping

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        docs = list(self._mapping.get(request.query, []))[: request.top_k]
        return RetrievalResult(docs=docs)


def _d(text: str, rank: int, *, cid: str | None = None) -> RetrievedDoc:
    return RetrievedDoc(text=text, score=1.0 / rank, rank=rank, chunk_id=cid)


def test_evaluate_retriever_aggregates_metrics_correctly():
    # Q1: один ref, есть в первом документе — recall@k=1, mrr=1, ndcg@3=1.
    # Q2: ref не встречается — нули.
    mapping = {
        "Q1": [_d("answer about reflection", 1), _d("noise", 2), _d("noise2", 3)],
        "Q2": [_d("totally different", 1), _d("noise", 2)],
    }
    retr = _CannedRetriever("test", mapping)
    dataset = [
        {"question": "Q1", "references": ["reflection"]},
        {"question": "Q2", "references": ["not present anywhere"]},
    ]
    out = evaluate_retriever(
        retr,
        dataset,
        k_values=[1, 5],
        fetch_k=10,
        hit_fns={"substring": substring_match_hit},
        primary_metric_mode="substring",
    )
    assert out["n_queries"] == 2
    # mean across two queries: (1+0)/2 = 0.5
    assert out["metrics"]["recall@1"] == pytest.approx(0.5)
    assert out["metrics"]["recall@5"] == pytest.approx(0.5)
    assert out["metrics"]["mrr"] == pytest.approx(0.5)
    assert "substring::recall@1" in out["metrics"]


def test_evaluate_retriever_uses_reference_quote_fallback():
    mapping = {
        "Q": [_d("the quote is here, exactly", 1)],
    }
    retr = _CannedRetriever("t", mapping)
    dataset = [
        {"question": "Q", "references": [], "reference_quote": "the quote is here"},
    ]
    out = evaluate_retriever(
        retr,
        dataset,
        k_values=[1],
        fetch_k=3,
        hit_fns={"substring": substring_match_hit},
        primary_metric_mode="substring",
    )
    assert out["metrics"]["recall@1"] == pytest.approx(1.0)


def test_evaluate_retriever_writes_trace_jsonl():
    mapping = {"Q": [_d("answer with reflection word", 1, cid="A")]}
    retr = _CannedRetriever("dense_test", mapping)
    dataset = [{"question": "Q", "references": ["reflection"]}]
    buf = io.StringIO()
    out = evaluate_retriever(
        retr,
        dataset,
        k_values=[1],
        fetch_k=3,
        hit_fns={"substring": substring_match_hit},
        primary_metric_mode="substring",
        trace_sink=buf,
        trace_doc_chars=100,
        trace_meta={"profile": "unit_test"},
    )
    buf.seek(0)
    lines = [l for l in buf.read().splitlines() if l]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["record"] == "query_trace"
    assert payload["retriever_name"] == "dense_test"
    assert payload["fetch_k"] == 3
    assert payload["profile"] == "unit_test"
    assert payload["candidates"][0]["chunk_id"] == "A"
    assert out["metrics"]["recall@1"] == pytest.approx(1.0)


def test_evaluate_retriever_multi_mode_keys():
    mapping = {"Q": [_d("alpha beta gamma", 1)]}
    retr = _CannedRetriever("t", mapping)
    dataset = [{"question": "Q", "references": ["alpha beta"]}]
    out = evaluate_retriever(
        retr,
        dataset,
        k_values=[1],
        fetch_k=3,
        hit_fns={
            "substring": substring_match_hit,
            "token_f1": token_overlap_f1_hit,
        },
        primary_metric_mode="substring",
    )
    # Both modes should produce hits and aggregated keys.
    assert out["metrics"]["substring::recall@1"] == pytest.approx(1.0)
    assert out["metrics"]["token_f1::recall@1"] == pytest.approx(1.0)

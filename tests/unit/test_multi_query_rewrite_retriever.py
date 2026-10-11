"""MultiQueryRewriteRetriever: парсер LLM-ответа + RRF по N перефразировкам."""
from __future__ import annotations

from typing import List, Optional

import pytest

from core.rag.retrievers.multi_query_rewrite_retriever import (
    MultiQueryRewriteRetriever,
    _parse_rewrites,
)
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)


class _PerQueryRetriever:
    """Возвращает заранее заданные docs в зависимости от текста запроса."""

    name = "perq"

    def __init__(self, mapping: dict[str, List[RetrievedDoc]]) -> None:
        self._mapping = mapping
        self.calls: List[str] = []

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request.query)
        return RetrievalResult(docs=list(self._mapping.get(request.query, []))[: request.top_k])


class _FixedLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str, temperature=None, max_tokens=None) -> str:
        self.last_prompt = prompt
        return self.response


class _FailingLLM:
    def complete(self, *a, **kw) -> str:
        raise RuntimeError("boom")


def _d(text: str, rank: int, *, cid: str | None = None) -> RetrievedDoc:
    return RetrievedDoc(text=text, score=1.0 / rank, rank=rank, chunk_id=cid)


def test_parse_rewrites_handles_numbered_and_bulleted():
    raw = (
        "1. Перефраз A\n"
        "2) Перефраз B\n"
        "- Перефраз C\n"
        "* Перефраз D\n"
        "Перефраз E\n"
        "1. Перефраз A\n"  # duplicate
    )
    out = _parse_rewrites(raw, expected_n=10)
    assert out == ["Перефраз A", "Перефраз B", "Перефраз C", "Перефраз D", "Перефраз E"]


def test_parse_rewrites_caps_to_expected_n():
    raw = "\n".join(f"{i}. q{i}" for i in range(10))
    assert _parse_rewrites(raw, expected_n=3) == ["q0", "q1", "q2"]


def test_parse_rewrites_ignores_empty_and_short_lines():
    raw = "1. a\n\n2.   \n3. b\n"
    assert _parse_rewrites(raw, expected_n=5) == ["a", "b"]


def test_mqr_runs_one_search_per_query_and_rrf_combines():
    mapping = {
        "orig?": [_d("doc-A", 1, cid="A"), _d("doc-B", 5, cid="B")],
        "alt-1": [_d("doc-B", 1, cid="B"), _d("doc-C", 3, cid="C")],
        "alt-2": [_d("doc-A", 2, cid="A"), _d("doc-D", 1, cid="D")],
    }
    base = _PerQueryRetriever(mapping)
    llm = _FixedLLM("1. alt-1\n2. alt-2\n")
    mqr = MultiQueryRewriteRetriever(
        base, llm, n_queries=2, rrf_k=60, include_original_query=True,
    )

    res = mqr.search(RetrievalRequest(query="orig?", top_k=4))

    # три вызова: original + 2 rewrites
    assert base.calls == ["orig?", "alt-1", "alt-2"]

    # 4 уникальных документа найдено
    assert {d.chunk_id for d in res.docs} == {"A", "B", "C", "D"}

    # A встречается в orig + alt-2 → высший RRF; B в orig + alt-1.
    a = next(d for d in res.docs if d.chunk_id == "A")
    b = next(d for d in res.docs if d.chunk_id == "B")
    c = next(d for d in res.docs if d.chunk_id == "C")
    assert a.score > c.score
    assert b.score > c.score
    assert a.metadata["mqr_sub_query_ranks"] == {0: 1, 2: 2}
    assert res.debug["mqr_n_rewrites_generated"] == 2


def test_mqr_falls_back_to_only_original_on_llm_error():
    mapping = {"q": [_d("x", 1, cid="X")]}
    base = _PerQueryRetriever(mapping)
    mqr = MultiQueryRewriteRetriever(base, _FailingLLM(), include_original_query=True)

    res = mqr.search(RetrievalRequest(query="q", top_k=1))

    assert base.calls == ["q"]
    assert res.debug["mqr_n_rewrites_generated"] == 0
    assert "boom" in (res.debug.get("mqr_llm_error") or "")
    assert res.docs[0].chunk_id == "X"


def test_mqr_can_exclude_original_query():
    mapping = {
        "orig": [_d("from-orig", 1, cid="O")],
        "alt-1": [_d("from-alt", 1, cid="A")],
    }
    base = _PerQueryRetriever(mapping)
    llm = _FixedLLM("1. alt-1\n")
    mqr = MultiQueryRewriteRetriever(
        base, llm, n_queries=1, include_original_query=False,
    )
    res = mqr.search(RetrievalRequest(query="orig", top_k=2))
    assert base.calls == ["alt-1"]
    assert {d.chunk_id for d in res.docs} == {"A"}

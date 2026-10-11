"""HydeQueryExpansionRetriever: LLM-генерация гипотетического документа + dense search."""
from __future__ import annotations

from typing import List, Optional

import pytest

from core.rag.retrievers.hyde_query_expansion_retriever import (
    HydeQueryExpansionRetriever,
)
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    RetrievedDoc,
)


class _RecordingRetriever:
    """Запоминает все ``request`` объекты для последующей проверки."""

    name = "rec"

    def __init__(self, docs: List[RetrievedDoc]) -> None:
        self._docs = docs
        self.calls: List[RetrievalRequest] = []

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        return RetrievalResult(docs=list(self._docs)[: request.top_k])


class _FixedLLM:
    def __init__(self, response: str = "Гипотетический ответ.") -> None:
        self.response = response
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str, temperature=None, max_tokens=None) -> str:
        self.last_prompt = prompt
        return self.response


class _FailingLLM:
    def complete(self, prompt: str, temperature=None, max_tokens=None) -> str:
        raise RuntimeError("LLM exploded")


def test_hyde_uses_hypothetical_text_as_query():
    base_docs = [
        RetrievedDoc(text="answer about reflection", score=0.9, rank=1, chunk_id="A"),
    ]
    base = _RecordingRetriever(base_docs)
    llm = _FixedLLM("Рефлексия — это умение оценивать свои действия и решения.")
    hyde = HydeQueryExpansionRetriever(base, llm, temperature=0.0, max_tokens=128)

    res = hyde.search(RetrievalRequest(query="что такое рефлексия?", top_k=3))

    # base_retriever получил гипотетический текст, а не исходный вопрос.
    assert base.calls, "underlying retriever was not called"
    used_q = base.calls[0].query
    assert "Рефлексия" in used_q
    assert res.docs[0].chunk_id == "A"
    assert res.debug["hyde_used_query_is_hypothetical"] is True
    assert res.debug["hyde_original_query"] == "что такое рефлексия?"
    assert res.debug["hyde_llm_error"] is None
    # Кандидаты помечены.
    assert res.docs[0].metadata["hyde_used_hypothetical"] is True


def test_hyde_falls_back_to_original_query_on_llm_error():
    base_docs = [RetrievedDoc(text="anything", score=0.5, rank=1, chunk_id="X")]
    base = _RecordingRetriever(base_docs)
    hyde = HydeQueryExpansionRetriever(base, _FailingLLM())

    res = hyde.search(RetrievalRequest(query="original question", top_k=1))

    assert base.calls[0].query == "original question"
    assert res.debug["hyde_used_query_is_hypothetical"] is False
    assert "LLM exploded" in (res.debug.get("hyde_llm_error") or "")
    assert res.docs[0].metadata["hyde_used_hypothetical"] is False


def test_hyde_falls_back_on_empty_llm_response():
    base = _RecordingRetriever([RetrievedDoc(text="t", score=0.0, rank=1)])
    hyde = HydeQueryExpansionRetriever(base, _FixedLLM(response="   \n  "))
    hyde.search(RetrievalRequest(query="q", top_k=1))
    assert base.calls[0].query == "q"


def test_hyde_prompt_contains_question_placeholder_substituted():
    base = _RecordingRetriever([])
    llm = _FixedLLM(response="ok")
    hyde = HydeQueryExpansionRetriever(base, llm)
    hyde.search(RetrievalRequest(query="zzz_marker_qq", top_k=1))
    assert llm.last_prompt is not None
    assert "zzz_marker_qq" in llm.last_prompt

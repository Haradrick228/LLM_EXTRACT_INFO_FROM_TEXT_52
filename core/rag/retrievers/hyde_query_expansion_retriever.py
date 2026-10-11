"""HyDE (Hypothetical Document Embeddings) ретривер.

Идея HyDE (Gao et al., 2022): вместо того чтобы кодировать запрос — заставляем
LLM **сгенерировать гипотетический ответ/документ** по запросу, и уже его
передаём в dense-ретривер. Сильнее работает при «слабых» запросах (один-два
слова) и в ситуациях, когда лексика запроса далека от лексики корпуса.

Обёртка не зависит от конкретного эмбеддера: оборачивает любой
:class:`Retriever`. Текст гипотетического документа пишется в
``RetrievalResult.debug``.

Параметры:

- ``base_retriever`` — любой dense/sparse/hybrid ретривер;
- ``llm_client`` — объект с методом ``complete(prompt: str, temperature=..., max_tokens=...) -> str``
  (например ``OllamaLLMClient`` или ``DeepSeekLLMClient`` из
  :mod:`core.graphrag.llm_provider`);
- ``prompt_template`` — формат с плейсхолдером ``{question}``;
- ``temperature``/``max_tokens`` — параметры генерации;
- ``include_original_query`` — если ``True``, ретривер делает **два** поиска
  (по оригиналу и по гипотетическому документу), результаты объединяются
  через простую вставку по best-rank (RRF делается отдельно через
  :class:`HybridRrfRetriever`).
"""
from __future__ import annotations

import time
from typing import Any, Optional, Protocol

from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)

_DEFAULT_HYDE_PROMPT_RU = (
    "Ты эксперт по русскоязычным образовательным материалам. "
    "Напиши развёрнутый абзац-ответ на следующий вопрос так, как если бы он "
    "встречался в учебнике или статье. Не задавай встречных вопросов, не пиши "
    "вводные фразы. Если не уверен в фактах — пиши обобщённо, но по сути.\n\n"
    "Вопрос: {question}\n\n"
    "Гипотетический фрагмент:"
)


class _SupportsComplete(Protocol):  # pragma: no cover - typing only
    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str: ...


class HydeQueryExpansionRetriever:
    """HyDE-обёртка над любым :class:`Retriever`."""

    def __init__(
        self,
        base_retriever: Retriever,
        llm_client: _SupportsComplete,
        *,
        prompt_template: str = _DEFAULT_HYDE_PROMPT_RU,
        temperature: float = 0.0,
        max_tokens: int = 256,
        name: str = "hyde",
    ) -> None:
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.name = name

    def _expand(self, question: str) -> str:
        prompt = self.prompt_template.format(question=question)
        out = self.llm_client.complete(
            prompt, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return (out or "").strip()

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        try:
            hypothetical = self._expand(request.query)
        except Exception as exc:
            # LLM упал — fallback на исходный запрос.
            hypothetical = ""
            llm_error: Optional[str] = repr(exc)
        else:
            llm_error = None

        used_query = hypothetical or request.query
        sub_req = RetrievalRequest(query=used_query, top_k=request.top_k, extra=request.extra)
        result = self.base_retriever.search(sub_req)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        debug = dict(result.debug)
        debug.update(
            {
                "hyde_original_query": request.query,
                "hyde_hypothetical_text": hypothetical,
                "hyde_used_query_is_hypothetical": bool(hypothetical),
                "hyde_llm_error": llm_error,
                "hyde_prompt_chars": len(self.prompt_template),
                "hyde_max_tokens": self.max_tokens,
                "hyde_temperature": self.temperature,
                "hyde_total_ms": elapsed_ms,
            }
        )
        # rewriting timings without touching base ones
        timings = dict(result.timings_ms)
        timings.setdefault("hyde_total_ms", elapsed_ms)

        # Промаркируем кандидатов, что они пришли из HyDE-запроса, для трасс.
        marked_docs: list[RetrievedDoc] = []
        for d in result.docs:
            md = dict(d.metadata)
            md["hyde_used_hypothetical"] = bool(hypothetical)
            marked_docs.append(
                RetrievedDoc(
                    text=d.text,
                    score=d.score,
                    rank=d.rank,
                    chunk_id=d.chunk_id,
                    source=d.source,
                    metadata=md,
                )
            )
        return RetrievalResult(docs=marked_docs, timings_ms=timings, debug=debug)


__all__ = ["HydeQueryExpansionRetriever"]

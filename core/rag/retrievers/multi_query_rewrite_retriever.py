"""Multi-Query Rewriting (MQR) ретривер.

LLM генерирует ``N`` перефразировок исходного запроса, для каждой выполняется
поиск через ``base_retriever``, и результаты объединяются через RRF. Это
позволяет «выходить» за лексические границы исходного запроса и поднимает
recall на коротких/неточных формулировках.

Парсер ответа LLM устойчивый: принимает строки, начинающиеся с цифр (``1.``,
``2)``), маркеров (``-``, ``*``) или просто построчно. Пустые строки и дубликаты
игнорируются. Если LLM вернул меньше ``N`` пунктов — используем что есть,
если больше — берём первые ``N``.

Если ``include_original_query=True`` — оригинальный запрос **также** участвует
в RRF (рекомендуется).
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Protocol

from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)

_DEFAULT_MQR_PROMPT_RU = (
    "Ты помогаешь искать релевантные фрагменты в русскоязычном корпусе "
    "(книги, статьи, документы). Сформулируй {n} разнообразных перефразировок "
    "следующего вопроса, чтобы покрыть разные формулировки, синонимы и углы "
    "взгляда. Не дублируй вопрос дословно. Каждую перефразировку — на новой "
    "строке, нумеруя 1. 2. 3. ... Без комментариев и предисловий.\n\n"
    "Исходный вопрос: {question}\n\n"
    "Перефразировки:"
)

_LIST_LINE_RE = re.compile(r"^\s*(?:\d+\s*[.)\-]\s*|[\-*•]\s+)?(.+?)\s*$")


class _SupportsComplete(Protocol):  # pragma: no cover - typing only
    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str: ...


def _doc_key(d: RetrievedDoc) -> str:
    if d.chunk_id:
        return f"cid::{d.chunk_id}"
    if d.source:
        return f"src::{d.source}::{(d.text or '')[:64]}"
    return f"txt::{(d.text or '')[:128]}"


def _parse_rewrites(text: str, *, expected_n: int) -> List[str]:
    """Распарсить список перефразировок, устойчиво к нумерации/маркерам."""
    out: List[str] = []
    seen: set[str] = set()
    for raw_line in (text or "").splitlines():
        m = _LIST_LINE_RE.match(raw_line)
        if not m:
            continue
        line = m.group(1).strip().strip('"').strip("«»")
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= expected_n:
            break
    return out


class MultiQueryRewriteRetriever:
    """Multi-Query Rewriting + RRF поверх любого :class:`Retriever`."""

    def __init__(
        self,
        base_retriever: Retriever,
        llm_client: _SupportsComplete,
        *,
        n_queries: int = 4,
        rrf_k: int = 60,
        temperature: float = 0.3,
        max_tokens: int = 256,
        prompt_template: str = _DEFAULT_MQR_PROMPT_RU,
        include_original_query: bool = True,
        per_query_top_k: Optional[int] = None,
        name: str = "mqr",
    ) -> None:
        if n_queries <= 0:
            raise ValueError("n_queries must be >= 1")
        self.base_retriever = base_retriever
        self.llm_client = llm_client
        self.n_queries = int(n_queries)
        self.rrf_k = int(rrf_k)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.prompt_template = prompt_template
        self.include_original_query = bool(include_original_query)
        self.per_query_top_k = per_query_top_k
        self.name = name

    def _generate_rewrites(self, question: str) -> List[str]:
        prompt = self.prompt_template.format(question=question, n=self.n_queries)
        out = self.llm_client.complete(
            prompt, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return _parse_rewrites(out, expected_n=self.n_queries)

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        try:
            rewrites = self._generate_rewrites(request.query)
        except Exception as exc:
            rewrites = []
            llm_error: Optional[str] = repr(exc)
        else:
            llm_error = None

        queries: List[str] = []
        if self.include_original_query:
            queries.append(request.query)
        queries.extend(rewrites)
        if not queries:
            queries = [request.query]

        per_top = self.per_query_top_k or max(request.top_k, 20)

        accum: Dict[str, float] = {}
        best: Dict[str, RetrievedDoc] = {}
        sub_ranks: Dict[str, Dict[int, int]] = {}
        timings: Dict[str, int] = {}

        for q_idx, q in enumerate(queries):
            sub_req = RetrievalRequest(query=q, top_k=per_top, extra=request.extra)
            r = self.base_retriever.search(sub_req)
            for tk, tv in r.timings_ms.items():
                timings[f"sub_{q_idx}.{tk}"] = tv
            for d in r.docs:
                key = _doc_key(d)
                accum[key] = accum.get(key, 0.0) + 1.0 / (self.rrf_k + d.rank)
                if key not in best or d.rank < best[key].rank:
                    best[key] = d
                sub_ranks.setdefault(key, {})[q_idx] = d.rank

        order = sorted(accum.keys(), key=lambda k: accum[k], reverse=True)
        out_docs: List[RetrievedDoc] = []
        for new_rank, key in enumerate(order[: request.top_k], start=1):
            base = best[key]
            out_docs.append(
                RetrievedDoc(
                    text=base.text,
                    score=float(accum[key]),
                    rank=new_rank,
                    chunk_id=base.chunk_id,
                    source=base.source,
                    metadata={
                        **base.metadata,
                        "mqr_rrf_score": accum[key],
                        "mqr_sub_query_ranks": sub_ranks[key],
                        "mqr_n_queries_total": len(queries),
                    },
                )
            )

        elapsed = int((time.monotonic() - t0) * 1000)
        return RetrievalResult(
            docs=out_docs,
            timings_ms={**timings, "mqr_total_ms": elapsed},
            debug={
                "mqr_n_queries_total": len(queries),
                "mqr_n_rewrites_generated": len(rewrites),
                "mqr_used_queries": queries,
                "mqr_llm_error": llm_error,
                "mqr_rrf_k": self.rrf_k,
                "mqr_include_original_query": self.include_original_query,
                "mqr_temperature": self.temperature,
                "mqr_max_tokens": self.max_tokens,
            },
        )


__all__ = ["MultiQueryRewriteRetriever", "_parse_rewrites"]

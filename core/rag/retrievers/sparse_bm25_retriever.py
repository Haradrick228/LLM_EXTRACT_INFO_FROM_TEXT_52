"""BM25-ретривер по «зеркалу» Chroma-коллекции в виде sidecar-индекса.

Индекс собирается из JSONL-экспорта Chroma (см.
:mod:`eval.indexing.export_chroma_collection_to_jsonl`) и записывается в
``DATA/indexes/bm25/<collection_slug>/``:

- ``meta.json`` — версия, токенизатор, ``corpus_size``, ``built_unix``;
- ``corpus_ids.json`` — упорядоченный список ``chunk_id`` (или fallback uid);
- ``corpus_docs.json`` — соответствующий список текстов;
- ``corpus_sources.json`` — соответствующий список ``source`` (опц.);
- ``bm25.pkl`` — сериализованный ``rank_bm25.BM25Okapi``.

Использование::

    from core.rag.retrievers.sparse_bm25_retriever import (
        SparseBm25Retriever, build_bm25_index_from_jsonl,
    )
    build_bm25_index_from_jsonl(
        jsonl_path="DATA/runs/corpus_export_*/chunks_from_chroma.jsonl",
        out_dir="DATA/indexes/bm25/corpus_phase_b_bge_m3",
    )
    r = SparseBm25Retriever.from_dir("DATA/indexes/bm25/corpus_phase_b_bge_m3")
    res = r.search(RetrievalRequest(query="что такое рефлексия?", top_k=40))

Токенизация:

- lower + Unicode \\w+;
- (опц.) русская лемматизация через ``pymorphy3``, если установлена.

Эта реализация специально хранит индекс на диск, чтобы избежать переиндексации
при каждом запуске матрицы B3.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .retriever_base_types import RetrievalRequest, RetrievalResult, RetrievedDoc

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

INDEX_FILE_BM25 = "bm25.pkl"
INDEX_FILE_DOCS = "corpus_docs.json"
INDEX_FILE_IDS = "corpus_ids.json"
INDEX_FILE_SOURCES = "corpus_sources.json"
INDEX_FILE_META = "meta.json"


class _Tokenizer:
    """Дешёвый токенизатор: lower + \\w+; опционально pymorphy3 нормализация."""

    def __init__(self, *, use_morph: bool = False) -> None:
        self.use_morph = use_morph
        self._morph = None
        if use_morph:
            try:
                import pymorphy3  # type: ignore
                self._morph = pymorphy3.MorphAnalyzer()
            except Exception:
                self.use_morph = False

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        toks = [t.casefold() for t in _TOKEN_RE.findall(text)]
        if self.use_morph and self._morph is not None:
            try:
                toks = [self._morph.parse(t)[0].normal_form for t in toks]  # type: ignore[index]
            except Exception:
                return toks
        return toks


def build_bm25_index_from_jsonl(
    *,
    jsonl_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    text_field: str = "document",
    id_field: str = "id",
    source_field: str = "source",
    use_morph: bool = False,
) -> Dict[str, Any]:
    """Прочитать JSONL экспорта Chroma и собрать BM25 sidecar в ``out_dir``.

    Возвращает manifest (тот же словарь, что записан в ``meta.json``).
    """
    try:
        from rank_bm25 import BM25Okapi
    except Exception as exc:  # pragma: no cover - pip install rank-bm25
        raise RuntimeError(
            "rank-bm25 is required for SparseBm25Retriever; install via `pip install rank-bm25`"
        ) from exc

    jpath = Path(jsonl_path)
    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    tokenizer = _Tokenizer(use_morph=use_morph)
    corpus_ids: List[str] = []
    corpus_docs: List[str] = []
    corpus_sources: List[str] = []
    corpus_tokens: List[List[str]] = []

    with jpath.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get(text_field) or "").strip()
            if not text:
                continue
            cid = row.get(id_field) or row.get("chunk_id") or f"uid_{line_no:08d}"
            meta = row.get("metadata") or {}
            src = row.get(source_field) or (meta.get(source_field) if isinstance(meta, dict) else "") or ""
            corpus_ids.append(str(cid))
            corpus_docs.append(text)
            corpus_sources.append(str(src))
            corpus_tokens.append(tokenizer.tokenize(text))

    bm25 = BM25Okapi(corpus_tokens)
    (odir / INDEX_FILE_BM25).write_bytes(pickle.dumps(bm25))
    (odir / INDEX_FILE_IDS).write_text(json.dumps(corpus_ids, ensure_ascii=False), encoding="utf-8")
    (odir / INDEX_FILE_DOCS).write_text(json.dumps(corpus_docs, ensure_ascii=False), encoding="utf-8")
    (odir / INDEX_FILE_SOURCES).write_text(json.dumps(corpus_sources, ensure_ascii=False), encoding="utf-8")

    meta = {
        "version": 1,
        "built_unix": int(time.time()),
        "source_jsonl": str(jpath.resolve()),
        "corpus_size": len(corpus_docs),
        "tokenizer": "regex_w+casefold" + ("+pymorphy3" if tokenizer.use_morph else ""),
        "use_morph": bool(tokenizer.use_morph),
    }
    (odir / INDEX_FILE_META).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


class SparseBm25Retriever:
    """BM25 retriever, реализующий :class:`Retriever`."""

    def __init__(
        self,
        *,
        bm25: Any,
        corpus_ids: List[str],
        corpus_docs: List[str],
        corpus_sources: List[str],
        tokenizer: _Tokenizer,
        name: str = "sparse_bm25",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._bm25 = bm25
        self._corpus_ids = corpus_ids
        self._corpus_docs = corpus_docs
        self._corpus_sources = corpus_sources
        self._tokenizer = tokenizer
        self.name = name
        self.meta = meta or {}

    @classmethod
    def from_dir(cls, index_dir: str | os.PathLike[str], *, name: str = "sparse_bm25") -> "SparseBm25Retriever":
        d = Path(index_dir)
        meta_path = d / INDEX_FILE_META
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids = json.loads((d / INDEX_FILE_IDS).read_text(encoding="utf-8"))
        docs = json.loads((d / INDEX_FILE_DOCS).read_text(encoding="utf-8"))
        sources_path = d / INDEX_FILE_SOURCES
        sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else [""] * len(ids)
        bm25 = pickle.loads((d / INDEX_FILE_BM25).read_bytes())
        tok = _Tokenizer(use_morph=bool(meta.get("use_morph", False)))
        return cls(
            bm25=bm25,
            corpus_ids=ids,
            corpus_docs=docs,
            corpus_sources=sources,
            tokenizer=tok,
            name=name,
            meta=meta,
        )

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        tokens = self._tokenizer.tokenize(request.query)
        if not tokens:
            return RetrievalResult(docs=[], timings_ms={"bm25_query_ms": 0})
        scores = self._bm25.get_scores(tokens)

        # argsort убывающе, top-k
        k = min(request.top_k, len(scores))
        if k <= 0:
            return RetrievalResult(docs=[], timings_ms={"bm25_query_ms": 0})
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: List[RetrievedDoc] = []
        for rank, idx in enumerate(order, start=1):
            out.append(
                RetrievedDoc(
                    text=self._corpus_docs[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    chunk_id=self._corpus_ids[idx],
                    source=self._corpus_sources[idx] if idx < len(self._corpus_sources) else None,
                    metadata={"bm25_index_meta": self.meta},
                )
            )
        elapsed = int((time.monotonic() - t0) * 1000)
        return RetrievalResult(
            docs=out,
            timings_ms={"bm25_query_ms": elapsed},
            debug={"bm25_corpus_size": len(self._corpus_docs)},
        )


__all__ = [
    "SparseBm25Retriever",
    "build_bm25_index_from_jsonl",
    "INDEX_FILE_BM25",
    "INDEX_FILE_DOCS",
    "INDEX_FILE_IDS",
    "INDEX_FILE_SOURCES",
    "INDEX_FILE_META",
]

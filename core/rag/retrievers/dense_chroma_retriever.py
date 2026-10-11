"""Dense ретривер поверх Chroma.

Тонкая обёртка над ``chromadb.Collection.query(...)`` ровно с теми же эвристиками,
что используются сейчас в ``eval.retrieval.run_retrieval_eval`` (3 кандидата
``include`` для разных версий Chroma). Возвращает :class:`RetrievalResult`
с привычной нумерацией ``rank=1..n``.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

from .retriever_base_types import RetrievalRequest, RetrievalResult, RetrievedDoc


_INCLUDE_CANDIDATES: tuple[List[str], ...] = (
    ["documents", "distances", "metadatas"],
    ["documents", "distances"],
    ["documents"],
)


def _meta_chunk_id(meta: Optional[dict]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    for key in ("chunk_document_id", "chunk_id", "id", "source_id"):
        v = meta.get(key)
        if v:
            return str(v)
    return None


def _meta_source(meta: Optional[dict]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    v = meta.get("source")
    return str(v) if v else None


class DenseChromaRetriever:
    """Dense поверх ``chromadb.Collection``.

    Параметры:

    - ``collection`` — уже инициализированная Chroma-коллекция;
    - ``name`` — человекочитаемое имя для логов/трасс.

    Использование::

        from eval.chroma_utils import chroma_http_client, get_collection
        client = chroma_http_client(host, port)
        col = get_collection(client, "corpus_phase_b_bge_m3", "BAAI/bge-m3")
        r = DenseChromaRetriever(col, name="dense_bge_m3")
        res = r.search(RetrievalRequest(query="что такое рефлексия?", top_k=40))
    """

    def __init__(self, collection: Any, *, name: str = "dense_chroma") -> None:
        self.collection = collection
        self.name = name

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        last_err: Optional[BaseException] = None
        pack: Optional[dict] = None
        used_include: List[str] = []
        for inc in _INCLUDE_CANDIDATES:
            try:
                pack = self.collection.query(
                    query_texts=[request.query], n_results=request.top_k, include=inc,
                )
                used_include = inc
                break
            except Exception as exc:
                last_err = exc
        if pack is None:
            raise last_err if last_err else RuntimeError("Chroma query failed")
        elapsed = int((time.monotonic() - t0) * 1000)

        docs = (pack.get("documents") or [[]])[0]
        dists = (pack.get("distances") or [[]])[0] if "distances" in pack else []
        metas = (pack.get("metadatas") or [[]])[0] if "metadatas" in pack else []

        out: List[RetrievedDoc] = []
        for i, text in enumerate(docs):
            dist = dists[i] if i < len(dists) else None
            meta = metas[i] if i < len(metas) else None
            score = float(-dist) if isinstance(dist, (int, float)) else 0.0
            out.append(
                RetrievedDoc(
                    text=text or "",
                    score=score,
                    rank=i + 1,
                    chunk_id=_meta_chunk_id(meta),
                    source=_meta_source(meta),
                    metadata={
                        "chroma_distance": dist,
                        "chroma_include": used_include,
                        "raw_metadata": meta if isinstance(meta, dict) else {},
                    },
                )
            )
        return RetrievalResult(
            docs=out,
            timings_ms={"dense_chroma_query_ms": elapsed},
            debug={"chroma_include_used": used_include},
        )


__all__ = ["DenseChromaRetriever"]

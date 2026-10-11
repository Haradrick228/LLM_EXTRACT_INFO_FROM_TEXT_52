"""CrossEncoder/bi-encoder реранкер на базе ``sentence-transformers``.

Историческое имя: ``core.rag.rerank_registry``. Здесь живут две вещи:

1. ``load_reranker(model_name, mode, *, device)`` — низкоуровневая фабрика,
   возвращает ``(model_or_none, kind)`` где ``kind`` ∈ {``"cross"``, ``"bi"``, ``"none"``}.
2. ``build_reranker(model_name, mode, *, device, top_k)`` — высокоуровневая
   фабрика, возвращает объект, реализующий :class:`Reranker` protocol.

Эвристики выбора режима сохраняем теми же, что были в ``RagService``:
``RERANK_MODE=cross`` или ``cross-encoder`` в имени → CrossEncoder, иначе bi.
"""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Tuple

from common.hf_model_resolution import resolve_hf_model_path
from ..retrievers.retriever_base_types import RetrievedDoc
from .reranker_base_types import Reranker, RerankRequest, RerankResult

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[misc, assignment]
    SentenceTransformer = None  # type: ignore[misc, assignment]


def load_reranker(
    model_name: str,
    mode: str,
    *,
    device: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """Низкоуровневая фабрика. Возвращает ``(model_or_none, kind)``.

    ``kind`` ∈ {``"cross"``, ``"bi"``, ``"none"``}.
    """
    name = (model_name or "").strip()
    if not name:
        return None, "none"
    resolved_name = resolve_hf_model_path(name)

    mode_l = (mode or "auto").strip().lower()
    wants_cross = "cross-encoder" in name.lower() or mode_l == "cross"

    dev = (device if device is not None else os.getenv("RERANK_DEVICE") or "cuda").strip()
    ce_kw: dict[str, Any] = {}
    st_kw: dict[str, Any] = {}
    if dev and dev not in ("auto", "none"):
        ce_kw["device"] = dev
        st_kw["device"] = dev

    if wants_cross and CrossEncoder is not None:
        try:
            return CrossEncoder(resolved_name, **ce_kw), "cross"
        except TypeError:
            try:
                return CrossEncoder(resolved_name), "cross"
            except Exception:
                pass
        except Exception:
            pass

    if SentenceTransformer is None:
        return None, "none"
    try:
        return SentenceTransformer(resolved_name, **st_kw), "bi"
    except TypeError:
        try:
            return SentenceTransformer(resolved_name), "bi"
        except Exception:
            return None, "none"
    except Exception:
        return None, "none"


class CrossEncoderReranker(Reranker):
    """Реранкер на основе :class:`sentence_transformers.CrossEncoder`.

    Если ``model_name`` указывает на bi-encoder (или Cross не загрузился) —
    fallback на cosine между эмбеддингом запроса и документов.
    """

    def __init__(self, model_name: str, *, mode: str = "cross", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.mode = mode
        self._model, self._kind = load_reranker(model_name, mode, device=device)
        self.name = f"reranker::{self._kind}::{model_name}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, request: RerankRequest) -> RerankResult:
        docs = list(request.docs)
        if not docs:
            return RerankResult(docs=[], timings_ms=0)
        if self._model is None:
            return RerankResult(docs=docs[: request.top_k], timings_ms=0)

        t0 = time.monotonic()
        texts = [d.text or "" for d in docs]
        try:
            if self._kind == "cross":
                pairs = [(request.query, t) for t in texts]
                raw = self._model.predict(pairs)
                scores: List[float] = list(raw.tolist() if hasattr(raw, "tolist") else raw)
            else:
                q_emb = self._model.encode([request.query], convert_to_numpy=True, normalize_embeddings=True)[0]
                d_emb = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                scores = (d_emb @ q_emb).tolist()
        except Exception:
            return RerankResult(docs=docs[: request.top_k], timings_ms=int((time.monotonic() - t0) * 1000))

        scored = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        reranked: List[RetrievedDoc] = []
        for new_rank, (doc, score) in enumerate(scored[: request.top_k], start=1):
            reranked.append(
                RetrievedDoc(
                    text=doc.text,
                    score=float(score),
                    rank=new_rank,
                    chunk_id=doc.chunk_id,
                    source=doc.source,
                    metadata={**doc.metadata, "rerank_from_rank": doc.rank, "rerank_kind": self._kind},
                )
            )
        return RerankResult(docs=reranked, timings_ms=int((time.monotonic() - t0) * 1000))


def build_reranker(
    model_name: str,
    *,
    mode: str = "cross",
    device: Optional[str] = None,
) -> CrossEncoderReranker:
    """Высокоуровневая фабрика. Возвращает реранкер, соответствующий протоколу."""
    return CrossEncoderReranker(model_name, mode=mode, device=device)


__all__ = [
    "CrossEncoderReranker",
    "build_reranker",
    "load_reranker",
]

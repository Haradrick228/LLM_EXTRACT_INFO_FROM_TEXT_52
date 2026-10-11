"""BGE-M3 multi-vector ретривер с late interaction.

BGE-M3 (Chen et al., 2024) выдаёт **три** представления одновременно:

- ``dense`` — обычное dense-embedding (1024-d);
- ``sparse`` (lexical_weights) — ``{token_id -> weight}``; считается как
  «обогащённый BM25»;
- ``colbert`` (multi-vec) — пер-токенные эмбеддинги для late interaction
  (MaxSim).

Этот класс **переранжирует** пул кандидатов (``candidate_pool_retriever``)
тройной оценкой. Веса ``(w_dense, w_sparse, w_colbert)`` задаются (BGE-M3
README рекомендует ``0.4 / 0.2 / 0.4``). Если вес = 0 — компонент пропускается
(в т.ч. ColBERT для экономии VRAM).

Для тестов кодер абстрагирован протоколом :class:`MultiVectorEncoder`; в
production используется :class:`FlagEmbeddingMultiVectorEncoder` поверх
``FlagEmbedding.BGEM3FlagModel``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol

from .retriever_base_types import (
    RetrievalRequest,
    RetrievalResult,
    Retriever,
    RetrievedDoc,
)


class MultiVectorEncoder(Protocol):  # pragma: no cover - typing
    """Минимальный протокол для тестируемости BGE-M3 ретривера."""

    def encode_queries(self, queries: List[str]) -> Dict[str, Any]: ...
    def encode_documents(self, documents: List[str]) -> Dict[str, Any]: ...


def bgem3_dense_score(q_vec: Any, d_vec: Any) -> float:
    """Cosine между двумя dense-векторами (предполагается нормированными)."""
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for BGE-M3 scoring") from exc
    q = np.asarray(q_vec, dtype=np.float32)
    d = np.asarray(d_vec, dtype=np.float32)
    if q.size == 0 or d.size == 0:
        return 0.0
    qn = float(np.linalg.norm(q)) or 1.0
    dn = float(np.linalg.norm(d)) or 1.0
    return float(np.dot(q, d) / (qn * dn))


def bgem3_sparse_score(
    q_sparse: Dict[Any, float], d_sparse: Dict[Any, float]
) -> float:
    """Lexical-weight dot product (по пересечению ключей).

    ``q_sparse``/``d_sparse``: ``{token_id_or_str -> weight}``. Ключи могут
    приходить как ``int`` или ``str`` — нормализуем к ``str``.
    """
    if not q_sparse or not d_sparse:
        return 0.0
    q_norm = {str(k): float(v) for k, v in q_sparse.items()}
    total = 0.0
    for k, v in d_sparse.items():
        kw = q_norm.get(str(k))
        if kw is None:
            continue
        total += float(v) * kw
    return float(total)


def bgem3_colbert_score(q_tokens: Any, d_tokens: Any) -> float:
    """MaxSim между токенными эмбеддингами запроса и документа (ColBERT-style).

    Для каждого токена запроса берётся максимум cos-sim с любым токеном
    документа, далее суммируются и нормируются на ``len(q_tokens)``.
    """
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for ColBERT scoring") from exc
    q = np.asarray(q_tokens, dtype=np.float32)
    d = np.asarray(d_tokens, dtype=np.float32)
    if q.size == 0 or d.size == 0 or q.ndim != 2 or d.ndim != 2:
        return 0.0
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    d_norm[d_norm == 0] = 1.0
    q_unit = q / q_norm
    d_unit = d / d_norm
    sim = q_unit @ d_unit.T  # (T_q, T_d)
    return float(sim.max(axis=1).sum() / max(1, q.shape[0]))


class BgeM3MultiVectorRetriever:
    """Переранжирование пула кандидатов dense+sparse+ColBERT late interaction."""

    def __init__(
        self,
        candidate_pool_retriever: Retriever,
        encoder: MultiVectorEncoder,
        *,
        candidate_top_k: int = 100,
        w_dense: float = 0.4,
        w_sparse: float = 0.2,
        w_colbert: float = 0.4,
        name: str = "bge_m3_multi_vec",
    ) -> None:
        self.candidate_pool_retriever = candidate_pool_retriever
        self.encoder = encoder
        self.candidate_top_k = int(candidate_top_k)
        self.w_dense = float(w_dense)
        self.w_sparse = float(w_sparse)
        self.w_colbert = float(w_colbert)
        self.name = name

    def search(self, request: RetrievalRequest) -> RetrievalResult:
        t0 = time.monotonic()
        pool_req = RetrievalRequest(
            query=request.query,
            top_k=max(self.candidate_top_k, request.top_k),
            extra=request.extra,
        )
        pool = self.candidate_pool_retriever.search(pool_req)
        if not pool.docs:
            return RetrievalResult(
                docs=[],
                timings_ms={"bge_m3_total_ms": int((time.monotonic() - t0) * 1000)},
                debug={"reason": "empty_pool"},
            )

        doc_texts = [d.text or "" for d in pool.docs]
        t_enc_q = time.monotonic()
        q_pack = self.encoder.encode_queries([request.query])
        enc_q_ms = int((time.monotonic() - t_enc_q) * 1000)
        t_enc_d = time.monotonic()
        d_pack = self.encoder.encode_documents(doc_texts)
        enc_d_ms = int((time.monotonic() - t_enc_d) * 1000)

        def _safe_get_first(pack: Dict[str, Any], key: str, default: Any) -> Any:
            v = pack.get(key)
            if v is None:
                return default
            try:
                length = len(v)
            except TypeError:
                return default
            if length == 0:
                return default
            return v[0]

        q_dense = _safe_get_first(q_pack, "dense_vecs", None)
        q_sparse = _safe_get_first(q_pack, "lexical_weights", {})
        q_colbert = _safe_get_first(q_pack, "colbert_vecs", None)

        d_dense_raw = d_pack.get("dense_vecs")
        d_dense = list(d_dense_raw) if d_dense_raw is not None else []
        d_sparse_raw = d_pack.get("lexical_weights")
        d_sparse = list(d_sparse_raw) if d_sparse_raw is not None else [{} for _ in doc_texts]
        d_colbert_raw = d_pack.get("colbert_vecs")
        d_colbert = list(d_colbert_raw) if d_colbert_raw is not None else [None for _ in doc_texts]

        scored: List[tuple[int, float, float, float, float]] = []
        for i, d in enumerate(pool.docs):
            sd_ok = self.w_dense and q_dense is not None and i < len(d_dense) and d_dense[i] is not None
            sd = bgem3_dense_score(q_dense, d_dense[i]) if sd_ok else 0.0
            ss = (
                bgem3_sparse_score(q_sparse, d_sparse[i])
                if self.w_sparse and i < len(d_sparse)
                else 0.0
            )
            sc_ok = (
                self.w_colbert
                and q_colbert is not None
                and i < len(d_colbert)
                and d_colbert[i] is not None
            )
            sc = bgem3_colbert_score(q_colbert, d_colbert[i]) if sc_ok else 0.0
            fused = self.w_dense * sd + self.w_sparse * ss + self.w_colbert * sc
            scored.append((i, sd, ss, sc, fused))

        scored.sort(key=lambda x: x[4], reverse=True)
        out: List[RetrievedDoc] = []
        for new_rank, (i, sd, ss, sc, fused) in enumerate(scored[: request.top_k], start=1):
            base = pool.docs[i]
            out.append(
                RetrievedDoc(
                    text=base.text,
                    score=float(fused),
                    rank=new_rank,
                    chunk_id=base.chunk_id,
                    source=base.source,
                    metadata={
                        **base.metadata,
                        "bge_m3_dense_score": sd,
                        "bge_m3_sparse_score": ss,
                        "bge_m3_colbert_score": sc,
                        "bge_m3_fused_score": fused,
                        "bge_m3_weights": {
                            "dense": self.w_dense,
                            "sparse": self.w_sparse,
                            "colbert": self.w_colbert,
                        },
                        "bge_m3_rank_before_rerank": base.rank,
                    },
                )
            )

        elapsed = int((time.monotonic() - t0) * 1000)
        return RetrievalResult(
            docs=out,
            timings_ms={
                "bge_m3_total_ms": elapsed,
                "bge_m3_encode_query_ms": enc_q_ms,
                "bge_m3_encode_docs_ms": enc_d_ms,
                **{f"pool.{k}": v for k, v in pool.timings_ms.items()},
            },
            debug={
                "pool_size": len(pool.docs),
                "weights": {
                    "dense": self.w_dense,
                    "sparse": self.w_sparse,
                    "colbert": self.w_colbert,
                },
                "candidate_top_k": self.candidate_top_k,
            },
        )


class FlagEmbeddingMultiVectorEncoder:
    """Production-encoder поверх ``FlagEmbedding.BGEM3FlagModel``.

    Ленивая инициализация: модель грузится при первом вызове ``encode_*``.
    Подключение тяжёлое (≈2.3 GB VRAM), поэтому в unit-тестах используем
    стаб-кодер.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        use_fp16: bool = True,
        device: Optional[str] = None,
        batch_size: int = 12,
        max_query_length: int = 1024,
        max_passage_length: int = 8192,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = bool(use_fp16)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_query_length = int(max_query_length)
        self.max_passage_length = int(max_passage_length)
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "FlagEmbedding is required for FlagEmbeddingMultiVectorEncoder; "
                "install via `pip install FlagEmbedding`"
            ) from exc
        kwargs: Dict[str, Any] = {"use_fp16": self.use_fp16}
        if self.device:
            kwargs["device"] = self.device
        self._model = BGEM3FlagModel(self.model_name, **kwargs)
        return self._model

    def encode_queries(self, queries: List[str]) -> Dict[str, Any]:
        model = self._ensure_model()
        return model.encode(
            queries,
            batch_size=self.batch_size,
            max_length=self.max_query_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    def encode_documents(self, documents: List[str]) -> Dict[str, Any]:
        model = self._ensure_model()
        return model.encode(
            documents,
            batch_size=self.batch_size,
            max_length=self.max_passage_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )


__all__ = [
    "BgeM3MultiVectorRetriever",
    "FlagEmbeddingMultiVectorEncoder",
    "MultiVectorEncoder",
    "bgem3_dense_score",
    "bgem3_sparse_score",
    "bgem3_colbert_score",
]

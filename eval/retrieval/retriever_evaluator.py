"""Generic evaluator over :class:`core.rag.retrievers.Retriever`.

Сильно отличается от старого ``run_retrieval_eval.evaluate`` тем, что не
завязан на ``chromadb.Collection`` напрямую: любой объект, удовлетворяющий
протоколу :class:`core.rag.retrievers.retriever_base_types.Retriever`,
подойдёт — dense, BM25, RRF/linear/weighted hybrid, HyDE, BGE-M3 multi-vector
и т.д.

Возвращает:

- агрегированный плоский dict метрик (как у :func:`evaluate` в
  ``run_retrieval_eval``) для совместимости с UI/reporting;
- словарь ``metrics_by_mode`` (для отчёта/JSON).

Опционально пишет JSONL-трейс по каждому вопросу (с теми же стейджами, что и
старый evaluate, плюс новые поля ``rrf_score`` / ``rrf_source_ranks``).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional

from core.rag.metrics.ranking_aggregates import (
    mean_reciprocal_rank as _mrr_pluggable,
    ndcg_at_k as _ndcg_pluggable,
    precision_at_k as _precision_pluggable,
    recall_at_k as _recall_pluggable,
)
from core.rag.retrievers.retriever_base_types import (
    RetrievalRequest,
    Retriever,
)

HitFn = Callable[[str, str], bool]


def eval_progress_every_from_env() -> int:
    """Интервал логов ``[progress] … i/N`` по stderr (0 = выключено).

    ``RETRIEVAL_EVAL_PROGRESS_EVERY`` — целое >=1, либо 0/off/false для отключения.
    По умолчанию 5 (достаточно для длинных F2/F3/F4/F7 без спама на каждую строку).
    """
    raw = os.getenv("RETRIEVAL_EVAL_PROGRESS_EVERY", "5").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return 0
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def _emit_eval_progress(*, idx: int, total: int, prefix: str, every: int) -> None:
    if every <= 0 or total <= 0:
        return
    at = idx + 1
    if at == 1 or at % every == 0 or at == total:
        msg = f"[progress] {prefix}{at}/{total}" if prefix else f"[progress] {at}/{total}"
        print(msg, file=sys.stderr, flush=True)


def _refs_for_retrieval_metrics(row: Dict[str, Any]) -> List[str]:
    """См. одноимённую функцию в run_retrieval_eval — единый контракт."""
    raw = row.get("references")
    out: List[str] = []
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
    if out:
        return out
    rq = (row.get("reference_quote") or "").strip()
    if rq:
        return [rq]
    return []


def _doc_payload(text: str, max_chars: int) -> Dict[str, Any]:
    raw = text or ""
    if max_chars <= 0:
        return {"char_length": len(raw), "text_truncated": False, "text": raw}
    trunc = len(raw) > max_chars
    return {
        "char_length": len(raw),
        "text_truncated": trunc,
        "text": raw[:max_chars] if trunc else raw,
    }


def evaluate_retriever(
    retriever: Retriever,
    dataset: List[Dict[str, Any]],
    *,
    k_values: List[int],
    fetch_k: int,
    hit_fns: Dict[str, HitFn],
    primary_metric_mode: str,
    trace_sink: Optional[IO[str]] = None,
    trace_doc_chars: int = 2500,
    trace_meta: Optional[Dict[str, Any]] = None,
    progress_every: int = 0,
    progress_prefix: str = "",
) -> Dict[str, Any]:
    """Прогнать ``retriever`` по ``dataset`` и посчитать метрики.

    Возвращает ``{"metrics": {flat dict}, "metrics_by_mode": {...}, "n_queries": N}``.

    При ``progress_every > 0`` в stderr печатается ход по строкам датасета
    ``[progress] <prefix>row_i/total`` (1, каждые ``every``, последняя).
    """
    if not hit_fns:
        raise ValueError("hit_fns must not be empty")
    if primary_metric_mode not in hit_fns:
        primary_metric_mode = next(iter(hit_fns))

    metrics_by_mode: Dict[str, Dict[str, List[float]]] = {}
    for mode in hit_fns:
        per_mode: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
        per_mode.update({f"precision@{k}": [] for k in k_values})
        per_mode["mrr"] = []
        per_mode["ndcg@3"] = []
        metrics_by_mode[mode] = per_mode

    n_queries = 0
    total_rows = len(dataset)
    for idx, row in enumerate(dataset):
        _emit_eval_progress(
            idx=idx,
            total=total_rows,
            prefix=progress_prefix,
            every=progress_every,
        )
        q = row.get("question") or ""
        if not q.strip():
            continue
        refs = _refs_for_retrieval_metrics(row)

        t0 = time.monotonic()
        res = retriever.search(RetrievalRequest(query=q, top_k=fetch_k))
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        docs_texts: List[str] = [d.text or "" for d in res.docs]

        per_query_metrics: Dict[str, Dict[str, float]] = {}
        for mode, hit_fn in hit_fns.items():
            block: Dict[str, float] = {}
            for k in k_values:
                block[f"recall@{k}"] = _recall_pluggable(docs_texts, refs, k, hit_fn=hit_fn)
                block[f"precision@{k}"] = _precision_pluggable(docs_texts, refs, k, hit_fn=hit_fn)
            block["mrr"] = _mrr_pluggable(docs_texts, refs, hit_fn=hit_fn)
            block["ndcg@3"] = _ndcg_pluggable(docs_texts, refs, 3, hit_fn=hit_fn)
            per_query_metrics[mode] = block
            for key, val in block.items():
                metrics_by_mode[mode][key].append(val)

        if trace_sink is not None:
            payload: Dict[str, Any] = {
                **(trace_meta or {}),
                "record": "query_trace",
                "schema_version": 3,
                "component": "eval.retrieval.retriever_evaluator",
                "retriever_name": getattr(retriever, "name", "unknown"),
                "ts_unix": time.time(),
                "eval_row_index": idx,
                "question_text": q,
                "gold_references_count": len(refs),
                "fetch_k": fetch_k,
                "retrieval_timings_ms": dict(res.timings_ms),
                "retrieval_total_ms_wall": elapsed_ms,
                "retrieval_debug": dict(res.debug),
                "metrics_by_mode": per_query_metrics,
                "candidates": [
                    {
                        "rank": d.rank,
                        "score": d.score,
                        "chunk_id": d.chunk_id,
                        "source": d.source,
                        "metadata": d.metadata,
                        **_doc_payload(d.text, trace_doc_chars),
                    }
                    for d in res.docs
                ],
            }
            trace_sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
            trace_sink.flush()

        n_queries += 1

    flat: Dict[str, float] = {}
    primary = metrics_by_mode[primary_metric_mode]
    for k, vals in primary.items():
        flat[k] = sum(vals) / max(1, len(vals))
    for mode, per_mode in metrics_by_mode.items():
        for key, vals in per_mode.items():
            flat[f"{mode}::{key}"] = sum(vals) / max(1, len(vals))

    return {
        "metrics": flat,
        "metrics_by_mode": {
            mode: {key: (sum(vals) / max(1, len(vals))) for key, vals in pm.items()}
            for mode, pm in metrics_by_mode.items()
        },
        "n_queries": n_queries,
    }


def load_eval_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


__all__ = ["evaluate_retriever", "load_eval_jsonl", "HitFn", "eval_progress_every_from_env"]

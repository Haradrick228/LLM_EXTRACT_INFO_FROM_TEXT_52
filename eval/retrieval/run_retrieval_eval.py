import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Tuple

from common.environment_snapshot import file_sha256, git_head_sha
from common.hf_model_resolution import resolve_hf_model_path
from common.reproducibility_seeds import env_seed, set_global_seed
from core.rag.metrics.ranking_aggregates import (
    mean_reciprocal_rank as _mrr_pluggable,
    ndcg_at_k as _ndcg_pluggable,
    precision_at_k as _precision_pluggable,
    recall_at_k as _recall_pluggable,
)
from core.rag.metrics.substring_match_hit import substring_match_hit
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client, get_collection
from eval.reporting.log_eval_run_config_banner import write_eval_banner

from core.rag.rerankers.cross_encoder_reranker import load_reranker

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:  # optional dependency for progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def connect_client(
    embed_model: str,
    *,
    host: str,
    port: int,
    collection_name: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """``collection_name``: явное имя коллекции; ``None`` или пусто — ``CHROMA_COLLECTION`` из env, иначе ``default``."""
    client = chroma_http_client(host=host, port=port)
    if collection_name and collection_name.strip():
        col_name = collection_name.strip()
    else:
        col_name = os.getenv("CHROMA_COLLECTION", "default")
    if trust_remote_code:
        from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function

        resolved_embed_model = resolve_hf_model_path(embed_model)
        spec = {
            "backend": "sentence_transformers",
            "model_name": resolved_embed_model,
            "st_model_kwargs": {"trust_remote_code": True},
        }
        ef = build_chroma_embedding_function(spec)
        return client.get_collection(col_name, embedding_function=ef)
    return get_collection(client, col_name, embed_model)


def _effective_eval_shuffle(requested_shuffle: bool) -> bool:
    """Учитывает RETRIEVAL_EVAL_SHUFFLE при requested_shuffle=True (как для gold-матрицы)."""
    if not requested_shuffle:
        return False
    ev = os.getenv("RETRIEVAL_EVAL_SHUFFLE", "").strip().lower()
    if ev in ("0", "false", "no", "off"):
        return False
    if ev in ("1", "true", "yes", "on"):
        return True
    return True


def load_eval(path: Path, *, shuffle: bool = True) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if _effective_eval_shuffle(shuffle):
        random.shuffle(rows)
    return rows


def text_match_score(candidates: List[str], reference: str) -> float:
    ref = reference.lower()
    best = 0.0
    for cand in candidates:
        c = (cand or "").lower()
        if not c:
            continue
        if ref in c or c in ref:
            return 1.0
        overlap = len(set(ref.split()) & set(c.split()))
        if overlap:
            best = max(best, overlap / max(1, len(ref.split())))
    return best


def recall_at_k(found: List[str], refs: List[str], k: int) -> float:
    hits = 0
    for r in refs:
        if any(r.lower() in f.lower() or f.lower() in r.lower() for f in found[:k]):
            hits += 1
    return hits / max(1, len(refs))


def precision_at_k(found: List[str], refs: List[str], k: int) -> float:
    hits = 0
    for f in found[:k]:
        if any(r.lower() in f.lower() or f.lower() in r.lower() for r in refs):
            hits += 1
    return hits / max(1, k)


def mrr(found: List[str], refs: List[str]) -> float:
    refs_l = [r.lower() for r in refs]
    for i, f in enumerate(found):
        fl = f.lower()
        if any(r in fl or fl in r for r in refs_l):
            return 1.0 / (i + 1)
    return 0.0


def ndcg(found: List[str], refs: List[str], k: int) -> float:
    """Бинарная релевантность; DCG с делителем log2(rank+1)."""
    dcg = 0.0
    for i, f in enumerate(found[:k]):
        gain = 1.0 if any(r.lower() in f.lower() or f.lower() in r.lower() for r in refs) else 0.0
        if gain:
            dcg += gain / (1.0 if i == 0 else (math.log2(i + 2)))
    ideal_hits = min(len(refs), k)
    idcg = sum(1.0 / (1.0 if i == 0 else math.log2(i + 2)) for i in range(ideal_hits))
    if not idcg:
        return 0.0
    # Один и тот же ref может совпасть с несколькими top-документами; DCG суммирует по рангам, idcg — нет -> nDCG>1.
    return min(1.0, dcg / idcg)


def rerank(question: str, docs: List[str], model, mode: str) -> List[Tuple[str, float]]:
    if not model:
        return [(d, 0.0) for d in docs]
    try:
        if mode == "cross":
            pairs = [(question, d) for d in docs]
            raw = model.predict(pairs)
            scores = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        else:
            q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
            d_emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            scores = (d_emb @ q_emb).tolist()
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return scored
    except Exception:
        return [(d, 0.0) for d in docs]


def _maybe_torch_cuda_stats() -> str:
    """
    Best-effort CUDA memory stats for debugging OOM / stalls.
    Never raises.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return ""
        dev = torch.cuda.current_device()
        name = torch.cuda.get_device_name(dev)
        alloc = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        max_alloc = int(torch.cuda.max_memory_allocated(dev))
        max_reserved = int(torch.cuda.max_memory_reserved(dev))
        return (
            f"cuda_device={dev} name={name!r} "
            f"alloc_mb={alloc/1024/1024:.0f} reserved_mb={reserved/1024/1024:.0f} "
            f"max_alloc_mb={max_alloc/1024/1024:.0f} max_reserved_mb={max_reserved/1024/1024:.0f}"
        )
    except Exception:
        return ""


def _emit_progress(msg: str, sink: Optional[IO[str]]) -> None:
    print(msg, flush=True)
    if sink is not None:
        sink.write(msg + "\n")
        sink.flush()


def _question_fp(question: str) -> str:
    return hashlib.sha256((question or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def _trace_write(sink: IO[str], payload: Dict[str, Any]) -> None:
    sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sink.flush()


def _trace_doc_payload(doc: str, max_chars: int) -> Dict[str, Any]:
    raw = doc or ""
    n = len(raw)
    if max_chars <= 0:
        body = raw
        truncated = False
    else:
        truncated = n > max_chars
        body = raw[:max_chars] if truncated else raw
    return {"char_length": n, "text_truncated": truncated, "text": body}


def _rank_before_for_doc(pre_docs: List[str], doc: str) -> Tuple[Optional[int], bool]:
    """1-based rank in Chroma order; ambiguous if duplicate identical chunk texts."""
    hits = [i + 1 for i, d in enumerate(pre_docs) if d == doc]
    if not hits:
        return None, False
    return hits[0], len(hits) > 1


def _refs_for_retrieval_metrics(row: Dict[str, Any]) -> List[str]:
    """Тексты для recall/precision/MRR/nDCG: список ``references``; если пусто — ``reference_quote``."""
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


HitFn = Callable[[str, str], bool]


def _parse_metric_modes(raw: str) -> List[str]:
    """Распарсить ``--metric-modes`` из строки в список нормализованных имён."""
    allowed = {"substring", "token_f1", "semantic"}
    out: List[str] = []
    for part in (raw or "").split(","):
        s = part.strip().lower().replace("-", "_")
        if not s:
            continue
        if s not in allowed:
            raise ValueError(f"unknown metric mode {s!r}; allowed: {sorted(allowed)}")
        if s not in out:
            out.append(s)
    return out or ["substring"]


def _build_hit_functions(
    modes: List[str],
    *,
    semantic_model_name: Optional[str] = None,
    semantic_threshold: float = 0.62,
    semantic_device: Optional[str] = None,
) -> Dict[str, HitFn]:
    """Собрать словарь {имя_метрики: hit_fn}.

    ``semantic`` лениво подгружает sentence-transformers (см.
    :mod:`core.rag.metrics.semantic_similarity_hit`). При недоступности модели
    автоматически падает в substring-fallback (см. логику scorer'а).
    """
    out: Dict[str, HitFn] = {}
    for m in modes:
        if m == "substring":
            out[m] = substring_match_hit
        elif m == "token_f1":
            out[m] = token_overlap_f1_hit
        elif m == "semantic":
            from core.rag.metrics.semantic_similarity_hit import (
                build_semantic_similarity_scorer,
            )

            scorer = build_semantic_similarity_scorer(
                semantic_model_name or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                threshold=semantic_threshold,
                device=semantic_device,
            )
            out[m] = scorer.hit_fn
        else:
            raise ValueError(f"unknown metric mode {m!r}")
    return out


def evaluate(
    col,
    dataset: List[Dict],
    k_values: List[int],
    fetch_k: int,
    rerank_model=None,
    rerank_mode: str = "none",
    *,
    show_progress: bool = False,
    force_plain_progress: bool = False,
    progress_every: int = 5,
    progress_sink: Optional[IO[str]] = None,
    trace_sink: Optional[IO[str]] = None,
    trace_doc_chars: int = 2000,
    trace_meta: Optional[Dict[str, Any]] = None,
    hit_fns: Optional[Dict[str, HitFn]] = None,
    primary_metric_mode: str = "substring",
) -> Dict[str, float]:
    """Оценить retrieval. Считает метрики по каждой ``hit_fn`` параллельно.

    Возвращаемый плоский ``dict[str, float]`` содержит ключи:

    - "recall@{k}", "precision@{k}", "mrr", "ndcg@3" — по ``primary_metric_mode``
      (по умолчанию ``substring``, как раньше), для обратной совместимости;
    - "<mode>::recall@{k}", "<mode>::mrr", ... — для каждой метрики из ``hit_fns``.

    Если ``hit_fns`` не задано — используется только substring.
    """
    if hit_fns is None:
        hit_fns = {"substring": substring_match_hit}
    if primary_metric_mode not in hit_fns:
        primary_metric_mode = next(iter(hit_fns))

    metrics_by_mode: Dict[str, Dict[str, List[float]]] = {}
    for mode in hit_fns:
        per_mode: Dict[str, List[float]] = {f"recall@{k}": [] for k in k_values}
        per_mode.update({f"precision@{k}": [] for k in k_values})
        per_mode["mrr"] = []
        per_mode["ndcg@3"] = []
        metrics_by_mode[mode] = per_mode

    it = dataset
    if show_progress and (not force_plain_progress) and tqdm is not None:
        it = tqdm(dataset, desc="eval.run_retrieval_eval", unit="q")

    if show_progress and force_plain_progress:
        _emit_progress(
            f"[progress] start rows={len(dataset)} fetch_k={fetch_k} rerank={'on' if rerank_model else 'off'}",
            progress_sink,
        )

    for idx, row in enumerate(it):
        q = row["question"]
        refs = _refs_for_retrieval_metrics(row)
        try:
            include_candidates: List[List[str]] = [
                ["documents", "distances", "metadatas"],
                ["documents", "distances"],
                ["documents"],
            ]
            res = None
            last_err: Optional[BaseException] = None
            used_include: List[str] = []
            for inc in include_candidates:
                try:
                    res = col.query(query_texts=[q], n_results=fetch_k, include=inc)
                    used_include = inc
                    break
                except Exception as e:
                    last_err = e
            if res is None:
                raise last_err if last_err else RuntimeError("Chroma query failed")
            pack = res or {}
            docs_raw = pack.get("documents", [[]])
            pre_docs = [d or "" for d in (docs_raw[0] if docs_raw else [])]
            dist_pack = pack.get("distances") or [[]]
            dist_row = dist_pack[0] if dist_pack else []
            meta_pack = pack.get("metadatas") or [[]]
            meta_row = meta_pack[0] if meta_pack else []

            pre_trace: List[Dict[str, Any]] = []
            for i, doc in enumerate(pre_docs):
                dist = dist_row[i] if i < len(dist_row) else None
                cid = None
                if i < len(meta_row) and isinstance(meta_row[i], dict):
                    m = meta_row[i]
                    cid = (
                        m.get("chunk_document_id")
                        or m.get("id")
                        or m.get("chunk_id")
                        or m.get("source_id")
                    )
                    if cid is not None:
                        cid = str(cid)
                cell = {
                    "rank_after_chroma": i + 1,
                    "chunk_id": cid,
                    "chroma_distance": dist,
                    "chroma_include": used_include,
                }
                cell.update(_trace_doc_payload(doc, trace_doc_chars))
                pre_trace.append(cell)

            post_trace: List[Dict[str, Any]] = []
            scored_pairs: List[Tuple[str, float]] = []
            if rerank_model:
                scored_pairs = rerank(q, pre_docs, rerank_model, rerank_mode)
                for j, (doc, score) in enumerate(scored_pairs):
                    rb, amb = _rank_before_for_doc(pre_docs, doc)
                    cell = {
                        "rank_after_rerank": j + 1,
                        "reranker_score": float(score),
                        "rank_after_chroma_before_rerank": rb,
                        "match_ambiguous": amb,
                    }
                    cell.update(_trace_doc_payload(doc, trace_doc_chars))
                    post_trace.append(cell)
                docs = [d for d, _ in scored_pairs]
            else:
                docs = list(pre_docs)

            per_query_metrics: Dict[str, Dict[str, float]] = {}
            for mode, hit_fn in hit_fns.items():
                m_block: Dict[str, float] = {}
                for k in k_values:
                    m_block[f"recall@{k}"] = _recall_pluggable(
                        docs if rerank_model else pre_docs, refs, k, hit_fn=hit_fn,
                    )
                    m_block[f"precision@{k}"] = _precision_pluggable(
                        docs if rerank_model else pre_docs, refs, k, hit_fn=hit_fn,
                    )
                m_block["mrr"] = _mrr_pluggable(
                    docs if rerank_model else pre_docs, refs, hit_fn=hit_fn,
                )
                m_block["ndcg@3"] = _ndcg_pluggable(
                    docs if rerank_model else pre_docs, refs, 3, hit_fn=hit_fn,
                )
                per_query_metrics[mode] = m_block

            if trace_sink is not None:
                base = dict(trace_meta or {})
                trace_payload: Dict[str, Any] = {
                    **base,
                    "record": "query_trace",
                    "schema_version": 2,
                    "component": "eval.run_retrieval_eval",
                    "ts_unix": time.time(),
                    "eval_row_index": idx,
                    "question_fingerprint_sha256_16": _question_fp(q),
                    "question_text": q,
                    "gold_references_count": len(refs),
                    "metrics_by_mode": per_query_metrics,
                    "stage_01_dense_chroma_retrieval": {
                        "n_results_requested": fetch_k,
                        "n_candidates_returned": len(pre_docs),
                        "candidates_ordered_by_chroma": pre_trace,
                    },
                    "stage_02_rerank": None,
                }
                if rerank_model:
                    trace_payload["stage_02_rerank"] = {
                        "rerank_model": base.get("rerank_model"),
                        "rerank_mode": rerank_mode,
                        "rerank_device": base.get("rerank_device"),
                        "n_candidates_input": len(pre_docs),
                        "n_candidates_output": len(post_trace),
                        "candidates_ordered_after_rerank": post_trace,
                    }
                _trace_write(trace_sink, trace_payload)
        except RuntimeError as e:
            msg = str(e)
            if "out of memory" in msg.lower() or "cuda" in msg.lower():
                extra = _maybe_torch_cuda_stats()
                raise RuntimeError(f"CUDA OOM during retrieval/rerank at row={idx}: {msg}. {extra}".strip()) from e
            raise

        if show_progress and (force_plain_progress or tqdm is None) and progress_every > 0:
            if (idx + 1) == 1 or (idx + 1) % progress_every == 0:
                extra = _maybe_torch_cuda_stats()
                suffix = f" ({extra})" if extra else ""
                _emit_progress(f"[progress] {idx+1}/{len(dataset)}{suffix}", progress_sink)

        for mode, hit_fn in hit_fns.items():
            per_mode = metrics_by_mode[mode]
            for k in k_values:
                per_mode[f"recall@{k}"].append(
                    _recall_pluggable(docs, refs, k, hit_fn=hit_fn)
                )
                per_mode[f"precision@{k}"].append(
                    _precision_pluggable(docs, refs, k, hit_fn=hit_fn)
                )
            per_mode["mrr"].append(_mrr_pluggable(docs, refs, hit_fn=hit_fn))
            per_mode["ndcg@3"].append(_ndcg_pluggable(docs, refs, 3, hit_fn=hit_fn))

    out: Dict[str, float] = {}
    primary = metrics_by_mode[primary_metric_mode]
    for k, vals in primary.items():
        out[k] = sum(vals) / max(1, len(vals))
    for mode, per_mode in metrics_by_mode.items():
        for key, vals in per_mode.items():
            out[f"{mode}::{key}"] = sum(vals) / max(1, len(vals))
    return out


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation for Chroma-backed RAG.")
    parser.add_argument("--eval-file", required=True, help="Path to eval_set.jsonl")
    parser.add_argument("--embed-model", default=os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    parser.add_argument("--k", default="5,10", help="Comma-separated k values, e.g. 5,10")
    parser.add_argument("--fetch-k", type=int, default=int(os.getenv("RETRIEVER_FETCH_K", "50") or 50), help="How many docs to fetch before rerank")
    parser.add_argument("--rerank-model", default=os.getenv("RERANK_MODEL", ""), help="Optional rerank model (CrossEncoder or bi-encoder)")
    parser.add_argument("--rerank-mode", default=os.getenv("RERANK_MODE", "auto"), help="auto|cross|bi")
    parser.add_argument("--out-json", default="", help="If set, write metrics JSON to this path")
    parser.add_argument("--progress", action="store_true", help="Show progress while evaluating (tqdm if available).")
    parser.add_argument(
        "--progress-plain",
        action="store_true",
        help="Print plain progress lines (disables tqdm; useful for logs/CI).",
    )
    parser.add_argument("--progress-every", type=int, default=5, help="If tqdm missing, print progress every N rows.")
    parser.add_argument(
        "--progress-log",
        default="",
        help="Append-friendly log path for plain progress lines (UTF-8). Useful when terminal capture drops output.",
    )
    parser.add_argument(
        "--chroma-host",
        default=os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1")),
        help="Хост Chroma для eval (по умолчанию localhost; RAG_EVAL_CHROMA_HOST переопределяет CHROMA_HOST из .env compose).",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000"))),
        help="Порт Chroma (проброс Docker, обычно 18000).",
    )
    parser.add_argument(
        "--chroma-collection",
        default="",
        help="Имя коллекции Chroma (пусто = CHROMA_COLLECTION из окружения, иначе default).",
    )
    parser.add_argument(
        "--embed-trust-remote-code",
        action="store_true",
        help="Для query-эмбеддера (SentenceTransformer в Chroma): trust_remote_code=True (нужно для jina-embeddings-v3 и др.).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("RETRIEVAL_EVAL_MAX_ROWS", "0") or 0),
        help="Ограничить число вопросов (0 = все). Можно задать через RETRIEVAL_EVAL_MAX_ROWS.",
    )
    parser.add_argument(
        "--retrieval-trace-jsonl",
        default="",
        help="JSONL: по одной JSON-строке на вопрос — этапы dense Chroma и реранк, чанки, scores, модели.",
    )
    parser.add_argument(
        "--trace-doc-chars",
        type=int,
        default=int(os.getenv("RETRIEVAL_TRACE_DOC_CHARS", "2500") or 2500),
        help="Максимум символов текста чанка в трейсе; 0 = без обрезки.",
    )
    parser.add_argument(
        "--trace-profile-label",
        default="",
        help="Метка сценария (например имя rag_profile) — дублируется в каждой записи трейса.",
    )
    parser.add_argument(
        "--metric-modes",
        default=os.getenv("RETRIEVAL_METRIC_MODES", "substring,token_f1"),
        help="Через запятую: substring,token_f1,semantic. Первый из перечисленных — primary.",
    )
    parser.add_argument(
        "--semantic-model",
        default=os.getenv(
            "RETRIEVAL_SEMANTIC_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
        help="Модель для semantic-similarity hit (только если в --metric-modes есть 'semantic').",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.62") or 0.62),
        help="Порог cosine для semantic hit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=env_seed(0, "EVAL_SEED"),
        help="Глобальный seed (EVAL_SEED). 0 по умолчанию.",
    )
    args = parser.parse_args()

    set_global_seed(int(args.seed))
    resolved_collection = (args.chroma_collection or "").strip() or os.getenv("CHROMA_COLLECTION", "default")
    chunking_hint = os.getenv("CHUNKING_LABEL", os.getenv("CHUNK_PROFILE", ""))
    trace_path = (args.retrieval_trace_jsonl or "").strip()
    shuffle_requested = not bool(trace_path)
    shuffle_rows = _effective_eval_shuffle(shuffle_requested)

    metric_modes = _parse_metric_modes(args.metric_modes)
    primary_metric_mode = metric_modes[0]
    hit_fns = _build_hit_functions(
        metric_modes,
        semantic_model_name=args.semantic_model,
        semantic_threshold=float(args.semantic_threshold),
        semantic_device=os.getenv("RERANK_DEVICE", None),
    )
    gold_sha = file_sha256(Path(args.eval_file))
    git_sha = git_head_sha(_ROOT)
    banner_fields: Dict[str, Any] = {
        "eval_file": args.eval_file,
        "embed_model": args.embed_model,
        "rerank_model": args.rerank_model or "(none)",
        "rerank_mode": args.rerank_mode,
        "rerank_device": os.getenv("RERANK_DEVICE", ""),
        "chroma_host": args.chroma_host,
        "chroma_port": args.chroma_port,
        "chroma_collection": resolved_collection,
        "fetch_k": args.fetch_k,
        "k": args.k,
        "max_rows": args.max_rows,
        "embed_trust_remote_code": str(bool(args.embed_trust_remote_code)),
        "chunking_label": chunking_hint or "(not set; see index manifest / reindex script)",
        "retrieval_trace_jsonl": trace_path or "(none)",
        "trace_doc_chars": args.trace_doc_chars,
        "eval_shuffle": str(shuffle_rows),
        "eval_shuffle_requested": str(shuffle_requested),
        "retrieval_eval_shuffle_env": os.getenv("RETRIEVAL_EVAL_SHUFFLE", "(unset)"),
        "seed": args.seed,
        "metric_modes": ",".join(metric_modes),
        "primary_metric_mode": primary_metric_mode,
        "gold_sha256": gold_sha or "(unavailable)",
        "git_sha": git_sha or "(unavailable)",
        "eval_scenario_type": os.getenv("EVAL_SCENARIO_TYPE", "").strip() or "(unset)",
    }

    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    dataset = load_eval(Path(args.eval_file), shuffle=shuffle_requested)
    if args.max_rows > 0:
        dataset = dataset[: args.max_rows]
    col = connect_client(
        args.embed_model,
        host=args.chroma_host,
        port=args.chroma_port,
        collection_name=(args.chroma_collection or "").strip() or None,
        trust_remote_code=bool(args.embed_trust_remote_code),
    )
    try:
        banner_fields["chroma_collection_count"] = int(col.count())
    except Exception:
        banner_fields["chroma_collection_count"] = "(unavailable)"
    write_eval_banner("run_retrieval_eval", banner_fields)

    device = os.getenv("RERANK_DEVICE", "cuda")
    rerank_model, rerank_mode = load_reranker(args.rerank_model, args.rerank_mode, device=device)
    show_progress = bool(args.progress)
    force_plain_progress = bool(args.progress_plain)

    trace_meta: Dict[str, Any] = {
        "trace_profile_label": (args.trace_profile_label or "").strip() or None,
        "chroma_collection": resolved_collection,
        "chroma_collection_count": banner_fields.get("chroma_collection_count"),
        "embed_model": args.embed_model,
        "rerank_model": (args.rerank_model or "").strip() or None,
        "rerank_device": device,
        "eval_scenario_type": banner_fields.get("eval_scenario_type"),
    }

    trace_sink: Optional[IO[str]] = None
    if trace_path:
        tp = Path(trace_path)
        tp.parent.mkdir(parents=True, exist_ok=True)
        trace_sink = tp.open("w", encoding="utf-8")
        _trace_write(
            trace_sink,
            {
                "record": "run_header",
                "schema_version": 1,
                "component": "eval.run_retrieval_eval",
                "ts_unix": time.time(),
                **trace_meta,
                "eval_file": args.eval_file,
                "max_rows": args.max_rows,
                "fetch_k": args.fetch_k,
                "k_list": k_values,
                "rerank_mode_resolved": rerank_mode,
                "eval_shuffle_disabled_for_trace": not shuffle_rows,
                "trace_doc_chars": args.trace_doc_chars,
                "trace_schema": (
                    "query_trace: stage_01_dense_chroma_retrieval.candidates_ordered_by_chroma = "
                    "rank_after_chroma, chunk_id, chroma_distance, char_length, text_truncated, text; "
                    "stage_02_rerank.candidates_ordered_after_rerank = rank_after_rerank, reranker_score, "
                    "rank_after_chroma_before_rerank, match_ambiguous, text."
                ),
            },
        )

    progress_sink: Optional[IO[str]] = None
    progress_path = (args.progress_log or "").strip()
    if progress_path:
        pl = Path(progress_path)
        pl.parent.mkdir(parents=True, exist_ok=True)
        progress_sink = pl.open("w", encoding="utf-8")

    _emit_progress(
        f"[progress] rerank_ready model={args.rerank_model!r} mode={rerank_mode!r} device={device!r} rows={len(dataset)}",
        progress_sink,
    )

    scores: Dict[str, float] = {}
    try:
        scores = evaluate(
            col,
            dataset,
            k_values,
            fetch_k=args.fetch_k,
            rerank_model=rerank_model,
            rerank_mode=rerank_mode,
            show_progress=show_progress,
            force_plain_progress=force_plain_progress,
            progress_every=max(1, int(args.progress_every)),
            progress_sink=progress_sink,
            trace_sink=trace_sink,
            trace_doc_chars=max(0, int(args.trace_doc_chars)),
            trace_meta=trace_meta,
            hit_fns=hit_fns,
            primary_metric_mode=primary_metric_mode,
        )
    finally:
        if trace_sink is not None:
            try:
                _trace_write(
                    trace_sink,
                    {
                        "record": "run_footer",
                        "schema_version": 1,
                        "component": "eval.run_retrieval_eval",
                        "ts_unix": time.time(),
                        "rows_evaluated": len(dataset),
                        "metrics_summary": scores,
                    },
                )
            except Exception:
                pass
            trace_sink.close()
        if progress_sink is not None:
            progress_sink.close()
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
    if args.out_json.strip():
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(scores)
        payload["_meta"] = {
            "eval_file": args.eval_file,
            "embed_model": args.embed_model,
            "fetch_k": args.fetch_k,
            "rerank_model": args.rerank_model,
            "rerank_mode": rerank_mode,
            "chroma_host": args.chroma_host,
            "chroma_port": args.chroma_port,
            "chroma_collection": resolved_collection,
            "max_rows": args.max_rows,
            "retrieval_trace_jsonl": trace_path or "",
            "trace_profile_label": (args.trace_profile_label or "").strip(),
            "eval_shuffle": shuffle_rows,
            "retrieval_eval_shuffle_env": os.getenv("RETRIEVAL_EVAL_SHUFFLE", ""),
            "seed": args.seed,
            "metric_modes": metric_modes,
            "primary_metric_mode": primary_metric_mode,
            "semantic_model": args.semantic_model if "semantic" in metric_modes else None,
            "semantic_threshold": float(args.semantic_threshold) if "semantic" in metric_modes else None,
            "gold_sha256": gold_sha,
            "git_sha": git_sha,
            "chroma_collection_count": banner_fields.get("chroma_collection_count"),
            "eval_scenario_type": banner_fields.get("eval_scenario_type"),
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", outp)


if __name__ == "__main__":
    main()

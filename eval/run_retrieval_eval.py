import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, IO, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client, get_collection

from core.rag.rerank_registry import load_reranker

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:  # optional dependency for progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def connect_client(embed_model: str, *, host: str, port: int):
    client = chroma_http_client(host=host, port=port)
    col_name = os.getenv("CHROMA_COLLECTION", "default")
    return get_collection(client, col_name, embed_model)


def load_eval(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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
    return dcg / idcg if idcg else 0.0


def rerank(question: str, docs: List[str], model, mode: str) -> List[Tuple[str, float]]:
    if not model:
        return [(d, 0.0) for d in docs]
    try:
        if mode == "cross":
            pairs = [(question, d) for d in docs]
            scores = model.predict(pairs).tolist()
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
) -> Dict[str, float]:
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics["mrr"] = []
    metrics["ndcg@3"] = []

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
        refs = row.get("references") or []
        try:
            res = col.query(query_texts=[q], n_results=fetch_k, include=["documents"])
            docs = (res or {}).get("documents", [[]])[0] if res else []
            docs = [d or "" for d in docs]
            if rerank_model:
                docs = [d for d, _ in rerank(q, docs, rerank_model, rerank_mode)]
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

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(docs, refs, k))
            metrics[f"precision@{k}"].append(precision_at_k(docs, refs, k))
        metrics["mrr"].append(mrr(docs, refs))
        metrics["ndcg@3"].append(ndcg(docs, refs, 3))

    out = {}
    for k, vals in metrics.items():
        out[k] = sum(vals) / max(1, len(vals))
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
        "--max-rows",
        type=int,
        default=int(os.getenv("RETRIEVAL_EVAL_MAX_ROWS", "0") or 0),
        help="Ограничить число вопросов (0 = все). Можно задать через RETRIEVAL_EVAL_MAX_ROWS.",
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    dataset = load_eval(Path(args.eval_file))
    if args.max_rows > 0:
        dataset = dataset[: args.max_rows]
    col = connect_client(args.embed_model, host=args.chroma_host, port=args.chroma_port)
    device = os.getenv("RERANK_DEVICE", "cuda")
    rerank_model, rerank_mode = load_reranker(args.rerank_model, args.rerank_mode, device=device)
    show_progress = bool(args.progress)
    force_plain_progress = bool(args.progress_plain)

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
        )
    finally:
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
            "chroma_collection": os.getenv("CHROMA_COLLECTION", "default"),
            "max_rows": args.max_rows,
        }
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote", outp)


if __name__ == "__main__":
    main()

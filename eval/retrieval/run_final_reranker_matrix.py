"""F7: финальная матрица реранкеров на победителе F2-F6.

«Победитель» (best retriever-конфиг) выбирается флагом ``--base``:

- ``dense:<scenario_base_id>`` — чистый dense из ``embedding_bases_full_corpus.yaml``;
- ``hybrid_rrf:<scenario_base_id>:<bm25_dir>`` — dense + BM25 RRF;
- ``bge_m3_multi:<scenario_base_id>:<combo>`` — BGE-M3 dense+sparse+ColBERT (для F4);
- ``hyde_mqr:<scenario_base_id>:<bm25_dir>`` — HyDE+MQR+dense+BM25 RRF (F3).

Реранкеры:

- ``--rerankers cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-v2-m3,jinaai/jina-reranker-v2-base-multilingual,Qwen/Qwen3-Reranker-4B``
  через запятую; режим определяется автоматически (cross|bi).

Для каждой пары ``(base, reranker)`` — отдельный JSON и trace.jsonl, плюс
сводный ``metrics_matrix.md``.

Запуск (smoke)::

    python -m eval.retrieval.run_final_reranker_matrix \\
        --base dense:full_mpnet \\
        --rerankers "cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-v2-m3" \\
        --candidate-top-k 50 --top-k 20 \\
        --max-rows 3
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.environment_snapshot import file_sha256, git_head_sha  # noqa: E402
from common.reproducibility_seeds import env_seed, set_global_seed  # noqa: E402
from core.rag.metrics.substring_match_hit import substring_match_hit  # noqa: E402
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit  # noqa: E402
from core.rag.rerankers.cross_encoder_reranker import build_reranker  # noqa: E402
from core.rag.retrievers.dense_chroma_retriever import DenseChromaRetriever  # noqa: E402
from core.rag.retrievers.hybrid_rrf_retriever import HybridRrfRetriever  # noqa: E402
from core.rag.retrievers.reranker_wrapping_retriever import (  # noqa: E402
    RerankerWrappingRetriever,
)
from core.rag.retrievers.sparse_bm25_retriever import SparseBm25Retriever  # noqa: E402
from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.experiments.phase_b_collections_registry import list_bases, load_registry  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402
from eval.retrieval.retriever_evaluator import evaluate_retriever, eval_progress_every_from_env, load_eval_jsonl  # noqa: E402

_REPORT_METRIC_KEYS = [
    "recall@5", "recall@10", "recall@20",
    "precision@5", "precision@10",
    "mrr", "ndcg@3",
]


def _merged_base(data: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(data.get("defaults") or {})
    m.update(base)
    return m


def _resolve_collection(names: set[str], primary: str, fallback: Optional[str]) -> str:
    if primary in names:
        return primary
    if fallback and fallback in names:
        return fallback
    raise ValueError(
        f"Chroma has neither collection={primary!r} nor fallback={fallback!r}"
    )


def _parse_base_spec(spec: str) -> Tuple[str, List[str]]:
    """``kind:arg1[:arg2[:...]]`` → (kind, [args])."""
    if ":" not in spec:
        raise ValueError(f"--base must be kind:args, got {spec!r}")
    parts = spec.split(":")
    kind = parts[0].strip()
    args = [p.strip() for p in parts[1:]]
    return kind, args


def _build_base_retriever(
    *,
    spec: str,
    reg: Dict[str, Any],
    chroma_names: set[str],
    host: str,
    port: int,
):
    kind, args = _parse_base_spec(spec)
    base_id = args[0] if args else ""
    bases = list_bases(reg)
    by_id = {str(_merged_base(reg, b)["base_id"]): _merged_base(reg, b) for b in bases}
    if base_id not in by_id:
        raise ValueError(f"base_id={base_id!r} not in registry")
    merged = by_id[base_id]
    resolved = _resolve_collection(chroma_names, str(merged["collection"]), merged.get("collection_fallback"))
    client = chroma_http_client(host=host, port=port)
    ef = build_chroma_embedding_function(merged)
    col = client.get_collection(resolved, embedding_function=ef)
    dense = DenseChromaRetriever(col, name=f"dense::{base_id}")

    if kind == "dense":
        return dense, {"kind": kind, "base_id": base_id, "collection": resolved}
    if kind == "hybrid_rrf":
        if len(args) < 2:
            raise ValueError("hybrid_rrf:<base_id>:<bm25_dir>")
        bm25_dir = Path(args[1])
        if not bm25_dir.is_dir():
            raise ValueError(f"bm25_dir not found: {bm25_dir}")
        bm25 = SparseBm25Retriever.from_dir(bm25_dir, name="bm25_corpus")
        rrf = HybridRrfRetriever([dense, bm25], k=60, per_retriever_top_k=200, name=f"hybrid_rrf::{base_id}")
        return rrf, {
            "kind": kind,
            "base_id": base_id,
            "collection": resolved,
            "bm25_dir": str(bm25_dir),
            "rrf_k": 60,
        }
    raise ValueError(
        f"Unknown --base kind={kind!r}. Supported: dense, hybrid_rrf. "
        f"For HyDE/MQR/BGE-M3 use the F3/F4 sweep scripts (артефакты можно реранкировать постфактум)."
    )


def _write_metrics_markdown(run_root: Path, summary: Dict[str, Any]) -> None:
    rerankers = summary["rerankers"]
    lines = [
        "# F7 final reranker matrix",
        "",
        f"Run directory: `{run_root}`",
        "",
        f"Base retriever: `{summary['base_meta']}`",
        f"candidate_top_k = `{summary['candidate_top_k']}`; top_k = `{summary['top_k']}`",
        "",
    ]
    for metric in _REPORT_METRIC_KEYS:
        lines.append(f"## {metric}")
        lines.append("")
        header = "| reranker | " + metric + " |"
        sep = "|---|---|"
        lines.extend([header, sep])
        for rname in rerankers:
            cell = summary["results"].get(rname, {})
            val = (cell.get("metrics") or {}).get(metric)
            lines.append(f"| {rname} | " + (f"{val:.4f}" if isinstance(val, (int, float)) else "") + " |")
        lines.append("")
    (run_root / "metrics_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def _rerank_name_slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace(" ", "_")


def _release_cuda_memory() -> None:
    """Освободить VRAM между реранкерами F7 (иначе все модели держатся в GPU сразу)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dense-registry", type=Path,
        default=_ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml",
    )
    p.add_argument("--base", required=True, help="kind:args; e.g. dense:full_mpnet, hybrid_rrf:full_mpnet:DATA/indexes/bm25/...")
    p.add_argument(
        "--rerankers",
        required=True,
        help="Comma-separated list of reranker model names.",
    )
    p.add_argument(
        "--rerank-mode-default",
        default="auto",
        help="auto|cross|bi (default auto: cross-encoder/* — cross, иначе bi).",
    )
    p.add_argument(
        "--rerank-device",
        default=os.getenv("RERANK_DEVICE", "cuda"),
    )
    p.add_argument(
        "--include-none-baseline",
        action="store_true",
        help="Добавить запись с reranker=None (только base) как baseline.",
    )
    p.add_argument(
        "--eval-file", type=Path,
        default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl",
    )
    p.add_argument("--candidate-top-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--k", default="5,10,20")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--metric-modes", default="substring,token_f1")
    p.add_argument(
        "--chroma-host",
        default=os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1")),
    )
    p.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000"))),
    )
    p.add_argument("--seed", type=int, default=env_seed(0, "EVAL_SEED"))
    p.add_argument("--run-tag", default="f7_final_reranker_matrix")
    args = p.parse_args()

    set_global_seed(int(args.seed))

    if not args.dense_registry.is_file():
        print(f"dense registry not found: {args.dense_registry}", file=sys.stderr)
        return 2
    if not args.eval_file.is_file():
        print(f"eval file not found: {args.eval_file}", file=sys.stderr)
        return 2

    reg = load_registry(args.dense_registry)

    chroma_names = {
        c.name
        for c in chroma_http_client(host=args.chroma_host, port=int(args.chroma_port)).list_collections()
    }

    try:
        base_retr, base_meta = _build_base_retriever(
            spec=args.base, reg=reg, chroma_names=chroma_names,
            host=args.chroma_host, port=int(args.chroma_port),
        )
    except Exception as exc:
        print(f"failed to build base retriever: {exc}", file=sys.stderr)
        return 2

    rerank_models: List[str] = [x.strip() for x in args.rerankers.split(",") if x.strip()]
    if not rerank_models:
        print("--rerankers must list at least one model", file=sys.stderr)
        return 2

    k_values: List[int] = [int(x) for x in args.k.split(",") if x.strip()]
    metric_modes: List[str] = [x.strip() for x in args.metric_modes.split(",") if x.strip()]
    primary_metric_mode = metric_modes[0]
    hit_fns: Dict[str, Any] = {}
    for m in metric_modes:
        if m == "substring":
            hit_fns[m] = substring_match_hit
        elif m == "token_f1":
            hit_fns[m] = token_overlap_f1_hit
        elif m == "semantic":
            from core.rag.metrics.semantic_similarity_hit import (
                build_semantic_similarity_scorer,
            )

            scorer = build_semantic_similarity_scorer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                threshold=float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.62")),
                device=os.getenv("RERANK_DEVICE"),
            )
            hit_fns[m] = scorer.hit_fn
        else:
            print(f"unknown metric mode: {m!r}", file=sys.stderr)
            return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_root = _ROOT / "eval" / "runs" / f"{ts}_{args.run_tag}"
    run_root.mkdir(parents=True, exist_ok=True)

    write_eval_banner(
        "eval.retrieval.run_final_reranker_matrix",
        {
            "run_dir": str(run_root.relative_to(_ROOT)),
            "base": args.base,
            "base_meta": json.dumps(base_meta, ensure_ascii=False),
            "rerankers": ",".join(rerank_models),
            "candidate_top_k": args.candidate_top_k,
            "top_k": args.top_k,
            "metric_modes": ",".join(metric_modes),
            "seed": args.seed,
            "gold_sha256": file_sha256(args.eval_file) or "(unavailable)",
            "git_sha": git_head_sha(_ROOT) or "(unavailable)",
        },
    )

    matrix_manifest = {
        "component": "eval.retrieval.run_final_reranker_matrix",
        "ts": ts,
        "base_spec": args.base,
        "base_meta": base_meta,
        "rerankers": rerank_models,
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "metric_modes": metric_modes,
        "primary_metric_mode": primary_metric_mode,
        "rerank_device": args.rerank_device,
        "seed": args.seed,
        "gold_sha256": file_sha256(args.eval_file),
        "git_sha": git_head_sha(_ROOT),
    }
    (run_root / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    dataset = load_eval_jsonl(args.eval_file)
    if args.max_rows > 0:
        dataset = dataset[: args.max_rows]

    rerankers_for_summary: List[str] = []
    summary: Dict[str, Any] = {
        "run_dir": str(run_root.relative_to(_ROOT)),
        "base_meta": base_meta,
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "n_queries": len(dataset),
        "rerankers": [],
        "results": {},
    }
    all_ok = True

    run_names: List[str] = []
    if args.include_none_baseline:
        run_names.append("__no_rerank__")
    run_names.extend(rerank_models)

    for rname in run_names:
        slug = _rerank_name_slug(rname)
        rerankers_for_summary.append(rname)
        reranker = None if rname == "__no_rerank__" else build_reranker(
            rname, mode=args.rerank_mode_default, device=args.rerank_device
        )
        wrap: Optional[RerankerWrappingRetriever] = None
        job_dir = run_root / slug
        job_dir.mkdir(exist_ok=True)
        trace_path = job_dir / f"{slug}.trace.jsonl"
        json_path = job_dir / f"{slug}.json"
        try:
            wrap = RerankerWrappingRetriever(
                base_retr,
                reranker,
                candidate_top_k=int(args.candidate_top_k),
                name=f"rerank({rname})::{base_meta.get('kind')}",
            )
            with trace_path.open("w", encoding="utf-8") as tsink:
                out = evaluate_retriever(
                    wrap,
                    dataset,
                    k_values=k_values,
                    fetch_k=int(args.top_k),
                    hit_fns=hit_fns,
                    primary_metric_mode=primary_metric_mode,
                    trace_sink=tsink,
                    trace_meta={
                        "base_spec": args.base,
                        "base_meta": base_meta,
                        "reranker": rname,
                        "candidate_top_k": int(args.candidate_top_k),
                    },
                    progress_every=eval_progress_every_from_env(),
                    progress_prefix=f"F7 {slug} ",
                )
            payload = {
                "_meta": {
                    "base_spec": args.base,
                    "base_meta": base_meta,
                    "reranker": rname,
                    "candidate_top_k": int(args.candidate_top_k),
                    "top_k": int(args.top_k),
                    "metric_modes": metric_modes,
                    "primary_metric_mode": primary_metric_mode,
                    "n_queries": out["n_queries"],
                },
                **out["metrics"],
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["results"][rname] = {
                "metrics": out["metrics"],
                "metrics_by_mode": out["metrics_by_mode"],
                "json": str(json_path.relative_to(_ROOT)),
                "trace": str(trace_path.relative_to(_ROOT)),
                "ok": True,
            }
        except Exception as exc:
            summary["results"][rname] = {"ok": False, "error": str(exc)}
            print(f"FAIL reranker={rname}: {exc!r}", file=sys.stderr)
            all_ok = False
        finally:
            wrap = None
            if reranker is not None:
                del reranker
                reranker = None
            _release_cuda_memory()
    summary["rerankers"] = rerankers_for_summary

    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _write_metrics_markdown(run_root, summary)
    print(f"Done. Artifacts in {run_root}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

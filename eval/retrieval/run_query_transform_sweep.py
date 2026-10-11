"""F3: HyDE + Multi-Query Rewriting (MQR) sweep.

Поверх dense-коллекций corpus_phase_b_* и BM25 sidecar — четыре ретривера на
сценарий:

1. ``dense_only`` — baseline (тот же, что в F2);
2. ``hyde`` — HyDE-обёртка над dense (LLM генерит гипотетический документ);
3. ``mqr`` — MQR-обёртка над dense (LLM генерит N перефразировок, RRF);
4. ``hyde_mqr_dense_bm25_rrf`` — финальный RRF поверх HyDE-dense, MQR-dense,
   dense_only, BM25.

LLM по умолчанию — локальный Ollama (qwen2.5:7b), через
:class:`core.graphrag.llm_provider.OllamaLLMClient`. При отсутствии Ollama
ретривер фолбэчит на оригинальный запрос (и трасса это записывает).

Запуск (smoke)::

    python -m eval.retrieval.run_query_transform_sweep \\
        --bm25-dir DATA/indexes/bm25/corpus_full_20260512 \\
        --scenarios full_mpnet \\
        --fetch-k 100 \\
        --eval-file eval/datasets/eval_set_gold_v2.jsonl \\
        --max-rows 3

Артефакты — те же, что у F2 (``summary.json``, ``metrics_matrix.md``,
``<scenario>/<job>/<job>.json`` и ``.trace.jsonl``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.environment_snapshot import file_sha256, git_head_sha  # noqa: E402
from common.reproducibility_seeds import env_seed, set_global_seed  # noqa: E402
from core.rag.metrics.substring_match_hit import substring_match_hit  # noqa: E402
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit  # noqa: E402
from core.rag.retrievers.dense_chroma_retriever import DenseChromaRetriever  # noqa: E402
from core.rag.retrievers.hybrid_rrf_retriever import HybridRrfRetriever  # noqa: E402
from core.rag.retrievers.hyde_query_expansion_retriever import (  # noqa: E402
    HydeQueryExpansionRetriever,
)
from core.rag.retrievers.multi_query_rewrite_retriever import (  # noqa: E402
    MultiQueryRewriteRetriever,
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


def _resolve_chroma_collection(names: set[str], primary: str, fallback: Optional[str]) -> str:
    if primary in names:
        return primary
    if fallback and fallback in names:
        return fallback
    raise ValueError(
        f"Chroma has neither collection={primary!r} nor fallback={fallback!r}"
    )


def _build_dense_retriever(
    merged: Dict[str, Any], resolved_collection: str, host: str, port: int
) -> DenseChromaRetriever:
    client = chroma_http_client(host=host, port=port)
    ef = build_chroma_embedding_function(merged)
    col = client.get_collection(resolved_collection, embedding_function=ef)
    return DenseChromaRetriever(col, name=f"dense::{merged.get('base_id', resolved_collection)}")


def _build_llm_client(provider: str, model: Optional[str]):
    """Lazily import + build LLM client. ``provider`` ∈ {ollama, deepseek}."""
    from core.graphrag.llm_provider import (
        DeepSeekLLMClient,
        OllamaLLMClient,
        build_llm_client,
    )

    pv = (provider or "").strip().lower()
    if pv == "ollama":
        return OllamaLLMClient(model=model)
    if pv == "deepseek":
        return DeepSeekLLMClient(model=model)
    return build_llm_client(model=model)


def _write_metrics_markdown(run_root: Path, summary: Dict[str, Any]) -> None:
    scenarios: List[str] = list(summary["scenarios"])
    jobs: List[str] = list(summary["jobs"])
    fk = summary["fetch_k"]
    lines = [
        "# F3 query-transform sweep",
        "",
        f"Run directory: `{run_root}`",
        "",
        f"fetch_k = `{fk}`; RRF k (фьюжн) = `{summary['rrf_k']}`",
        f"LLM provider: `{summary['llm_provider']}` model `{summary['llm_model']}`",
        "",
    ]
    for metric in _REPORT_METRIC_KEYS:
        lines.append(f"## {metric}")
        lines.append("")
        header = "| scenario | " + " | ".join(jobs) + " |"
        sep = "|" + "|".join(["---"] * (1 + len(jobs))) + "|"
        lines.extend([header, sep])
        for sid in scenarios:
            row_cells: List[str] = []
            for job in jobs:
                cell = summary["results"].get(sid, {}).get(job, {})
                val = (cell.get("metrics") or {}).get(metric)
                row_cells.append(f"{val:.4f}" if isinstance(val, (int, float)) else "")
            lines.append(f"| {sid} | " + " | ".join(row_cells) + " |")
        lines.append("")
    (run_root / "metrics_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dense-registry",
        type=Path,
        default=_ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml",
    )
    p.add_argument(
        "--scenarios",
        default="",
        help="Через запятую: base_id или 'all' (по умолчанию все, кроме full_default_clean).",
    )
    p.add_argument("--bm25-dir", type=Path, required=True)
    p.add_argument(
        "--eval-file",
        type=Path,
        default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl",
    )
    p.add_argument("--k", default="5,10,20")
    p.add_argument("--fetch-k", type=int, default=100, help="Pool size before metric.")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--metric-modes",
        default="substring,token_f1",
        help="Список через запятую (substring,token_f1,semantic).",
    )
    p.add_argument(
        "--llm-provider",
        default=os.getenv("RAG_QT_LLM_PROVIDER", "ollama"),
        help="ollama|deepseek|auto. auto = LLM_PROVIDER env.",
    )
    p.add_argument(
        "--llm-model",
        default=os.getenv("RAG_QT_LLM_MODEL", "qwen2.5:7b-instruct"),
    )
    p.add_argument("--hyde-temperature", type=float, default=0.0)
    p.add_argument("--hyde-max-tokens", type=int, default=256)
    p.add_argument("--mqr-n-queries", type=int, default=4)
    p.add_argument("--mqr-temperature", type=float, default=0.3)
    p.add_argument("--mqr-max-tokens", type=int, default=256)
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
    p.add_argument("--run-tag", default="f3_query_transform_sweep")
    args = p.parse_args()

    set_global_seed(int(args.seed))

    if not args.dense_registry.is_file():
        print(f"dense registry not found: {args.dense_registry}", file=sys.stderr)
        return 2
    if not args.bm25_dir.is_dir():
        print(f"BM25 sidecar not found: {args.bm25_dir}", file=sys.stderr)
        return 2
    if not args.eval_file.is_file():
        print(f"eval file not found: {args.eval_file}", file=sys.stderr)
        return 2

    reg = load_registry(args.dense_registry)
    bases_all = list_bases(reg)
    wanted_ids: Optional[set[str]] = None
    if args.scenarios.strip() and args.scenarios.strip().lower() != "all":
        wanted_ids = {x.strip() for x in args.scenarios.split(",") if x.strip()}

    scenarios: List[Dict[str, Any]] = []
    for b in bases_all:
        merged = _merged_base(reg, b)
        bid = str(merged["base_id"])
        if wanted_ids is not None and bid not in wanted_ids:
            continue
        if wanted_ids is None and bid == "full_default_clean":
            continue
        scenarios.append(merged)

    if not scenarios:
        print("No scenarios selected", file=sys.stderr)
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

    chroma_names = {
        c.name
        for c in chroma_http_client(host=args.chroma_host, port=int(args.chroma_port)).list_collections()
    }

    bm25 = SparseBm25Retriever.from_dir(args.bm25_dir, name="bm25_corpus")

    try:
        llm = _build_llm_client(args.llm_provider, args.llm_model)
    except Exception as exc:
        print(
            f"WARN: cannot build LLM client (provider={args.llm_provider}, "
            f"model={args.llm_model}): {exc!r}. Query transforms will fall back "
            f"to original query when used.",
            file=sys.stderr,
        )

        class _NullLLM:
            def complete(self, *a, **kw) -> str:
                raise RuntimeError("no LLM client (init failed)")

        llm = _NullLLM()

    write_eval_banner(
        "eval.retrieval.run_query_transform_sweep",
        {
            "run_dir": str(run_root.relative_to(_ROOT)),
            "dense_registry": str(args.dense_registry),
            "scenarios": ",".join(str(s["base_id"]) for s in scenarios),
            "bm25_dir": str(args.bm25_dir),
            "fetch_k": args.fetch_k,
            "rrf_k": args.rrf_k,
            "metric_modes": ",".join(metric_modes),
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model,
            "mqr_n_queries": args.mqr_n_queries,
            "eval_file": str(args.eval_file),
            "seed": args.seed,
            "gold_sha256": file_sha256(args.eval_file) or "(unavailable)",
            "git_sha": git_head_sha(_ROOT) or "(unavailable)",
        },
    )

    matrix_manifest: Dict[str, Any] = {
        "component": "eval.retrieval.run_query_transform_sweep",
        "ts": ts,
        "scenarios": [s["base_id"] for s in scenarios],
        "fetch_k": args.fetch_k,
        "rrf_k": args.rrf_k,
        "metric_modes": metric_modes,
        "primary_metric_mode": primary_metric_mode,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "mqr_n_queries": args.mqr_n_queries,
        "hyde_temperature": args.hyde_temperature,
        "hyde_max_tokens": args.hyde_max_tokens,
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

    jobs_order = ["dense_only", "hyde", "mqr", "hyde_mqr_dense_bm25_rrf"]

    summary: Dict[str, Any] = {
        "run_dir": str(run_root.relative_to(_ROOT)),
        "scenarios": [s["base_id"] for s in scenarios],
        "jobs": jobs_order,
        "fetch_k": args.fetch_k,
        "rrf_k": args.rrf_k,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "n_queries": len(dataset),
        "results": {},
    }

    all_ok = True
    for sc in scenarios:
        sid = str(sc["base_id"])
        col_yaml = str(sc["collection"])
        fb = sc.get("collection_fallback")
        try:
            resolved = _resolve_chroma_collection(
                chroma_names, col_yaml, fb if isinstance(fb, str) else None,
            )
            dense = _build_dense_retriever(sc, resolved, args.chroma_host, int(args.chroma_port))
        except Exception as exc:
            print(f"SKIP {sid}: {exc}", file=sys.stderr)
            summary["results"][sid] = {"error": str(exc)}
            all_ok = False
            continue

        hyde = HydeQueryExpansionRetriever(
            dense, llm,
            temperature=args.hyde_temperature,
            max_tokens=args.hyde_max_tokens,
            name=f"hyde::{sid}",
        )
        mqr = MultiQueryRewriteRetriever(
            dense, llm,
            n_queries=args.mqr_n_queries,
            rrf_k=args.rrf_k,
            temperature=args.mqr_temperature,
            max_tokens=args.mqr_max_tokens,
            include_original_query=True,
            name=f"mqr::{sid}",
        )
        fused = HybridRrfRetriever(
            [hyde, mqr, dense, bm25],
            k=args.rrf_k,
            per_retriever_top_k=args.fetch_k,
            name=f"hyde_mqr_dense_bm25_rrf::{sid}",
        )

        scen_dir = run_root / sid
        scen_dir.mkdir(parents=True, exist_ok=True)

        retrievers_by_job = {
            "dense_only": dense,
            "hyde": hyde,
            "mqr": mqr,
            "hyde_mqr_dense_bm25_rrf": fused,
        }

        summary["results"][sid] = {}
        for job in jobs_order:
            job_dir = scen_dir / job
            job_dir.mkdir(exist_ok=True)
            trace_path = job_dir / f"{job}.trace.jsonl"
            json_path = job_dir / f"{job}.json"
            try:
                with trace_path.open("w", encoding="utf-8") as tsink:
                    out = evaluate_retriever(
                        retrievers_by_job[job],
                        dataset,
                        k_values=k_values,
                        fetch_k=args.fetch_k,
                        hit_fns=hit_fns,
                        primary_metric_mode=primary_metric_mode,
                        trace_sink=tsink,
                        trace_meta={
                            "scenario": sid,
                            "job_name": job,
                            "fetch_k": args.fetch_k,
                            "rrf_k": int(args.rrf_k),
                            "llm_provider": args.llm_provider,
                            "llm_model": args.llm_model,
                        },
                        progress_every=eval_progress_every_from_env(),
                        progress_prefix=f"F3 {sid} {job} ",
                    )
                payload = {
                    "_meta": {
                        "scenario": sid,
                        "collection_yaml": col_yaml,
                        "collection_resolved": resolved,
                        "job_name": job,
                        "fetch_k": args.fetch_k,
                        "rrf_k": int(args.rrf_k),
                        "k_values": k_values,
                        "metric_modes": metric_modes,
                        "primary_metric_mode": primary_metric_mode,
                        "llm_provider": args.llm_provider,
                        "llm_model": args.llm_model,
                        "n_queries": out["n_queries"],
                    },
                    **out["metrics"],
                }
                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                summary["results"][sid][job] = {
                    "metrics": out["metrics"],
                    "metrics_by_mode": out["metrics_by_mode"],
                    "json": str(json_path.relative_to(_ROOT)),
                    "trace": str(trace_path.relative_to(_ROOT)),
                    "ok": True,
                }
            except Exception as exc:
                summary["results"][sid][job] = {"ok": False, "error": str(exc)}
                print(f"FAIL {sid} job={job}: {exc!r}", file=sys.stderr)
                all_ok = False

    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _write_metrics_markdown(run_root, summary)
    print(f"Done. Artifacts in {run_root}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

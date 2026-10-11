"""F2: BM25 + dense ⇒ RRF sweep по fetch_k {40, 100, 200, 400} на 5 dense-коллекциях.

Цель — вынуть «сколько релевантных чанков вообще можно вытащить» при заданном
пуле кандидатов. На вход — BM25 sidecar (общий на корпус) и
``embedding_bases_full_corpus.yaml`` (5 dense-коллекций). Для каждого
``(collection, fetch_k)`` считаются recall@k / precision@k / MRR / nDCG@3
тремя способами:

1. **dense-only** — baseline (см. ранние B3 прогоны);
2. **bm25-only** — sanity-check, ноль настроек;
3. **hybrid_rrf(k=60)** — Reciprocal Rank Fusion поверх двух выдач.

Запуск (smoke)::

    python -m eval.retrieval.run_hybrid_rrf_sweep \\
        --bm25-dir DATA/indexes/bm25/corpus_full_20260512 \\
        --fetch-ks 40,100,200,400 \\
        --eval-file eval/datasets/eval_set_gold_v2.jsonl \\
        --max-rows 3

Артефакты: ``eval/runs/<ts>_f2_hybrid_rrf_sweep/``:

- ``matrix_manifest.json`` — конфигурация прогона;
- ``summary.json`` — все ячейки матрицы (метрики + статус);
- ``metrics_matrix.md`` — markdown-таблицы по каждой метрике;
- ``<scenario>/<job>.json`` — детальный JSON метрик;
- ``<scenario>/<job>.trace.jsonl`` — JSONL-трейс по каждому вопросу
  (включая ``rrf_source_ranks`` и метаданные кандидатов).
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
from core.monitoring.experiment_run_context import experiment_run_context  # noqa: E402
from core.rag.metrics.substring_match_hit import substring_match_hit  # noqa: E402
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit  # noqa: E402
from core.rag.retrievers.dense_chroma_retriever import DenseChromaRetriever  # noqa: E402
from core.rag.retrievers.hybrid_rrf_retriever import HybridRrfRetriever  # noqa: E402
from core.rag.retrievers.sparse_bm25_retriever import SparseBm25Retriever  # noqa: E402
from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.experiments.phase_b_collections_registry import list_bases, load_registry  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402
from eval.retrieval.retriever_evaluator import (  # noqa: E402
    evaluate_retriever,
    eval_progress_every_from_env,
    load_eval_jsonl,
)

# Метрики, попадающие в matrix.md
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
        print(
            f"[hybrid] chroma_collection={primary!r} missing; using collection_fallback={fallback!r}",
            file=sys.stderr,
        )
        return fallback
    raise ValueError(
        f"Chroma has neither collection={primary!r} nor fallback={fallback!r}. "
        f"Existing (sample): {sorted(list(names))[:30]}"
    )


def _build_dense_retriever(
    merged: Dict[str, Any], resolved_collection: str, host: str, port: int
) -> DenseChromaRetriever:
    client = chroma_http_client(host=host, port=port)
    ef = build_chroma_embedding_function(merged)
    col = client.get_collection(resolved_collection, embedding_function=ef)
    return DenseChromaRetriever(col, name=f"dense::{merged.get('base_id', resolved_collection)}")


def _write_metrics_markdown(
    run_root: Path,
    summary: Dict[str, Any],
) -> None:
    scenarios: List[str] = list(summary["scenarios"])
    fetch_ks: List[int] = list(summary["fetch_ks"])
    job_kinds = ["dense_only", "bm25_only", "hybrid_rrf"]

    lines = [
        "# F2 hybrid retrieval sweep",
        "",
        f"Run directory: `{run_root}`",
        "",
        "Метрики по subsection — taken from `substring` hit_fn (primary), полный набор см. в `*.json`.",
        "",
        f"RRF k = `{summary['rrf_k']}`; fetch_k sweep = `{fetch_ks}`",
        "",
    ]

    for metric in _REPORT_METRIC_KEYS:
        lines.append(f"## {metric}")
        lines.append("")
        header = (
            "| scenario | fetch_k | "
            + " | ".join(job_kinds)
            + " |"
        )
        sep = "|" + "|".join(["---"] * (2 + len(job_kinds))) + "|"
        lines.extend([header, sep])
        for sid in scenarios:
            for fk in fetch_ks:
                cells: List[str] = []
                for job in job_kinds:
                    cell = (
                        summary["results"].get(sid, {})
                        .get(str(fk), {})
                        .get(job, {})
                    )
                    val = (cell.get("metrics") or {}).get(metric)
                    cells.append(f"{val:.4f}" if isinstance(val, (int, float)) else "")
                lines.append(f"| {sid} | {fk} | " + " | ".join(cells) + " |")
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
    p.add_argument(
        "--bm25-dir",
        type=Path,
        required=True,
        help="Каталог BM25 sidecar (см. eval.indexing.build_bm25_sidecar).",
    )
    p.add_argument(
        "--eval-file",
        type=Path,
        default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl",
    )
    p.add_argument("--k", default="5,10,20", help="comma-separated k values")
    p.add_argument(
        "--fetch-ks",
        default="40,100,200,400",
        help="comma-separated fetch_k sweep (pool size before metric).",
    )
    p.add_argument("--rrf-k", type=int, default=60, help="RRF rank-bias k (default 60).")
    p.add_argument("--max-rows", type=int, default=0, help="0 = use full eval-file.")
    p.add_argument(
        "--metric-modes",
        default="substring,token_f1",
        help="Список через запятую (substring,token_f1,semantic).",
    )
    p.add_argument(
        "--chroma-host", default=os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1")),
    )
    p.add_argument(
        "--chroma-port",
        type=int,
        default=int(os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000"))),
    )
    p.add_argument("--seed", type=int, default=env_seed(0, "EVAL_SEED"))
    p.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Не запускать BM25/RRF (для отладки dense-only базлайна).",
    )
    p.add_argument(
        "--run-tag",
        default="f2_hybrid_rrf_sweep",
        help="Слаг для каталога artifacts.",
    )
    args = p.parse_args()

    set_global_seed(int(args.seed))

    if not args.dense_registry.is_file():
        print(f"dense registry not found: {args.dense_registry}", file=sys.stderr)
        return 2
    if not args.bm25_dir.is_dir() and not args.skip_bm25:
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
            # Default-clean часто дубликат mpnet — по умолчанию пропускаем.
            continue
        scenarios.append(merged)

    if not scenarios:
        print("No scenarios selected", file=sys.stderr)
        return 2

    fetch_ks: List[int] = [int(x) for x in args.fetch_ks.split(",") if x.strip()]
    k_values: List[int] = [int(x) for x in args.k.split(",") if x.strip()]
    metric_modes: List[str] = [x.strip() for x in args.metric_modes.split(",") if x.strip()]
    primary_metric_mode = metric_modes[0]

    # Hit functions — substring + token_f1 без необходимости в HF-моделях.
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

    chroma_names: set[str] = set()
    if not args.skip_bm25 or any(True for _ in scenarios):
        try:
            chroma_names = {
                c.name
                for c in chroma_http_client(host=args.chroma_host, port=int(args.chroma_port)).list_collections()
            }
        except Exception as exc:
            print(f"WARN: cannot list Chroma collections: {exc!r}", file=sys.stderr)
            chroma_names = set()

    # BM25 retriever — общий на все сценарии.
    bm25: Optional[SparseBm25Retriever] = None
    if not args.skip_bm25:
        bm25 = SparseBm25Retriever.from_dir(args.bm25_dir, name="bm25_corpus")

    write_eval_banner(
        "eval.retrieval.run_hybrid_rrf_sweep",
        {
            "run_dir": str(run_root.relative_to(_ROOT)),
            "dense_registry": str(args.dense_registry),
            "scenarios": ",".join(str(s["base_id"]) for s in scenarios),
            "bm25_dir": str(args.bm25_dir),
            "fetch_ks": ",".join(str(x) for x in fetch_ks),
            "rrf_k": str(args.rrf_k),
            "eval_file": str(args.eval_file),
            "max_rows": args.max_rows,
            "metric_modes": ",".join(metric_modes),
            "seed": args.seed,
            "gold_sha256": file_sha256(args.eval_file) or "(unavailable)",
            "git_sha": git_head_sha(_ROOT) or "(unavailable)",
        },
    )

    matrix_manifest: Dict[str, Any] = {
        "component": "eval.retrieval.run_hybrid_rrf_sweep",
        "ts": ts,
        "dense_registry": str(args.dense_registry),
        "bm25_dir": str(args.bm25_dir),
        "eval_file": str(args.eval_file),
        "scenarios": [s["base_id"] for s in scenarios],
        "fetch_ks": fetch_ks,
        "rrf_k": args.rrf_k,
        "k_values": k_values,
        "metric_modes": metric_modes,
        "primary_metric_mode": primary_metric_mode,
        "seed": args.seed,
        "gold_sha256": file_sha256(args.eval_file),
        "git_sha": git_head_sha(_ROOT),
        "chroma_host": args.chroma_host,
        "chroma_port": int(args.chroma_port),
    }
    (run_root / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    dataset = load_eval_jsonl(args.eval_file)
    if args.max_rows > 0:
        dataset = dataset[: args.max_rows]

    summary: Dict[str, Any] = {
        "run_dir": str(run_root.relative_to(_ROOT)),
        "scenarios": [s["base_id"] for s in scenarios],
        "fetch_ks": fetch_ks,
        "rrf_k": args.rrf_k,
        "n_queries": len(dataset),
        "results": {},
    }

    all_ok = True
    for sc in scenarios:
        sid = str(sc["base_id"])
        col_yaml = str(sc["collection"])
        fb = sc.get("collection_fallback")
        try:
            resolved = _resolve_chroma_collection(chroma_names, col_yaml, fb if isinstance(fb, str) else None)
        except Exception as exc:
            print(f"SKIP {sid}: {exc}", file=sys.stderr)
            summary["results"][sid] = {"error": str(exc)}
            all_ok = False
            continue

        scen_dir = run_root / sid
        scen_dir.mkdir(parents=True, exist_ok=True)

        try:
            dense = _build_dense_retriever(sc, resolved, args.chroma_host, int(args.chroma_port))
        except Exception as exc:
            print(f"SKIP {sid}: dense build failed: {exc}", file=sys.stderr)
            summary["results"][sid] = {"error": f"dense build failed: {exc}"}
            all_ok = False
            continue

        summary["results"][sid] = {}
        for fk in fetch_ks:
            cell_root = scen_dir / f"fetch_k_{fk}"
            cell_root.mkdir(exist_ok=True)
            sub: Dict[str, Any] = {}
            jobs: List[tuple[str, Any]] = [("dense_only", dense)]
            if bm25 is not None:
                hybrid = HybridRrfRetriever(
                    [dense, bm25], k=int(args.rrf_k), per_retriever_top_k=fk,
                    name=f"hybrid_rrf::{sid}::fk{fk}",
                )
                jobs.extend([("bm25_only", bm25), ("hybrid_rrf", hybrid)])

            for job_name, retr in jobs:
                trace_path = cell_root / f"{job_name}.trace.jsonl"
                json_path = cell_root / f"{job_name}.json"
                try:
                    with trace_path.open("w", encoding="utf-8") as tsink:
                        out = evaluate_retriever(
                            retr,
                            dataset,
                            k_values=k_values,
                            fetch_k=fk,
                            hit_fns=hit_fns,
                            primary_metric_mode=primary_metric_mode,
                            trace_sink=tsink,
                            trace_meta={
                                "scenario": sid,
                                "fetch_k": fk,
                                "job_name": job_name,
                                "rrf_k": int(args.rrf_k),
                            },
                            progress_every=eval_progress_every_from_env(),
                            progress_prefix=f"F2 {sid} fk={fk} {job_name} ",
                        )
                    payload = {
                        "_meta": {
                            "scenario": sid,
                            "collection_yaml": col_yaml,
                            "collection_resolved": resolved,
                            "job_name": job_name,
                            "fetch_k": fk,
                            "rrf_k": int(args.rrf_k),
                            "k_values": k_values,
                            "metric_modes": metric_modes,
                            "primary_metric_mode": primary_metric_mode,
                            "n_queries": out["n_queries"],
                        },
                        **out["metrics"],
                    }
                    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    sub.setdefault(str(fk), {})[job_name] = {
                        "metrics": out["metrics"],
                        "metrics_by_mode": out["metrics_by_mode"],
                        "json": str(json_path.relative_to(_ROOT)),
                        "trace": str(trace_path.relative_to(_ROOT)),
                        "ok": True,
                    }
                except Exception as exc:
                    sub.setdefault(str(fk), {})[job_name] = {"ok": False, "error": str(exc)}
                    print(f"FAIL {sid} fk={fk} job={job_name}: {exc!r}", file=sys.stderr)
                    all_ok = False
            summary["results"][sid].update(sub)

    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _write_metrics_markdown(run_root, summary)
    print(f"Done. Artifacts in {run_root}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

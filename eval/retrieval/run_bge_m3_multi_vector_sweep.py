"""F4: BGE-M3 native dense+sparse+ColBERT sweep по весам fusion.

Применяет :class:`BgeM3MultiVectorRetriever` поверх dense-пула из Chroma
(коллекция ``corpus_phase_b_bge_m3``). На каждый запрос пересчитываются три
скоринга — dense, sparse(lexical_weights), ColBERT (MaxSim) — и фьюзятся как
``w_dense*d + w_sparse*s + w_colbert*c``.

В sweep сравниваются комбинации весов:

- ``dense_only`` (1.0/0.0/0.0) — baseline (что и так есть в Chroma),
- ``dense_sparse`` (0.6/0.4/0.0) — без ColBERT, бюджетно,
- ``dense_colbert`` (0.5/0.0/0.5) — late interaction без sparse,
- ``balanced`` (0.4/0.2/0.4) — рекомендация BGE-M3 README,
- ``sparse_heavy`` (0.2/0.6/0.2) — для лексических запросов.

Запуск (smoke)::

    python -m eval.retrieval.run_bge_m3_multi_vector_sweep \\
        --candidate-top-k 100 \\
        --eval-file eval/datasets/eval_set_gold_v2.jsonl \\
        --max-rows 3

Артефакты — каталог ``eval/runs/<ts>_f4_bge_m3_multi_vector_sweep/`` с
``summary.json``, ``metrics_matrix.md``, ``<combo>/<combo>.json|trace.jsonl``.

Production-кодер: :class:`FlagEmbeddingMultiVectorEncoder` поверх
``BAAI/bge-m3`` (требуется ``pip install FlagEmbedding``). Подгружается
лениво.
"""
from __future__ import annotations

import argparse
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
from core.rag.retrievers.bge_m3_multi_vector_retriever import (  # noqa: E402
    BgeM3MultiVectorRetriever,
    FlagEmbeddingMultiVectorEncoder,
)
from core.rag.retrievers.dense_chroma_retriever import DenseChromaRetriever  # noqa: E402
from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.experiments.phase_b_collections_registry import list_bases, load_registry  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402
from eval.retrieval.retriever_evaluator import evaluate_retriever, eval_progress_every_from_env, load_eval_jsonl  # noqa: E402

_DEFAULT_COMBOS: List[Tuple[str, Tuple[float, float, float]]] = [
    ("dense_only",    (1.0, 0.0, 0.0)),
    ("dense_sparse",  (0.6, 0.4, 0.0)),
    ("dense_colbert", (0.5, 0.0, 0.5)),
    ("balanced",      (0.4, 0.2, 0.4)),
    ("sparse_heavy",  (0.2, 0.6, 0.2)),
]

_REPORT_METRIC_KEYS = [
    "recall@5", "recall@10", "recall@20",
    "precision@5", "precision@10",
    "mrr", "ndcg@3",
]


def _merged_base(data: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(data.get("defaults") or {})
    m.update(base)
    return m


def _find_bge_m3_scenario(reg: Dict[str, Any]) -> Dict[str, Any]:
    for b in list_bases(reg):
        merged = _merged_base(reg, b)
        if str(merged.get("base_id")) == "full_bge_m3":
            return merged
    raise RuntimeError("base_id=full_bge_m3 not in registry")


def _parse_combos(raw: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    if not raw or raw.strip().lower() == "default":
        return list(_DEFAULT_COMBOS)
    out: List[Tuple[str, Tuple[float, float, float]]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"combo spec must be name=wd,ws,wc — got {part!r}")
        name, weights = part.split("=", 1)
        parts = [p.strip() for p in weights.split(",")]
        if len(parts) != 3:
            raise ValueError(f"combo spec must have 3 weights — got {weights!r}")
        out.append((name.strip(), (float(parts[0]), float(parts[1]), float(parts[2]))))
    return out


def _write_metrics_markdown(run_root: Path, summary: Dict[str, Any]) -> None:
    combos: List[Dict[str, Any]] = summary["combos"]
    lines = [
        "# F4 BGE-M3 multi-vector sweep",
        "",
        f"Run directory: `{run_root}`",
        "",
        f"Candidate pool top-k = `{summary['candidate_top_k']}`; metric pool top-k = `{summary['top_k']}`",
        "",
        "## Weights",
        "",
        "| combo | w_dense | w_sparse | w_colbert |",
        "|---|---|---|---|",
    ]
    for c in combos:
        lines.append(
            f"| {c['name']} | {c['w_dense']} | {c['w_sparse']} | {c['w_colbert']} |"
        )
    lines.append("")
    for metric in _REPORT_METRIC_KEYS:
        lines.append(f"## {metric}")
        lines.append("")
        header = "| combo | " + metric + " |"
        sep = "|---|---|"
        lines.extend([header, sep])
        for c in combos:
            cell = summary["results"].get(c["name"], {})
            val = (cell.get("metrics") or {}).get(metric)
            lines.append(f"| {c['name']} | " + (f"{val:.4f}" if isinstance(val, (int, float)) else "") + " |")
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
        "--eval-file",
        type=Path,
        default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl",
    )
    p.add_argument("--candidate-top-k", type=int, default=100)
    p.add_argument("--top-k", type=int, default=20, help="metric top-k after rerank.")
    p.add_argument("--k", default="5,10,20")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--metric-modes",
        default="substring,token_f1",
    )
    p.add_argument(
        "--combos",
        default="default",
        help="default | name=wd,ws,wc;...",
    )
    p.add_argument(
        "--bge-model", default="BAAI/bge-m3",
    )
    p.add_argument("--bge-fp16", action="store_true", default=True)
    p.add_argument("--bge-batch-size", type=int, default=12)
    p.add_argument(
        "--bge-device",
        default=os.getenv("RAG_BGE_M3_DEVICE", os.getenv("ST_EMBED_DEVICE")),
    )
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
    p.add_argument("--run-tag", default="f4_bge_m3_multi_vector_sweep")
    args = p.parse_args()

    set_global_seed(int(args.seed))

    if not args.dense_registry.is_file():
        print(f"dense registry not found: {args.dense_registry}", file=sys.stderr)
        return 2
    if not args.eval_file.is_file():
        print(f"eval file not found: {args.eval_file}", file=sys.stderr)
        return 2

    reg = load_registry(args.dense_registry)
    sc = _find_bge_m3_scenario(reg)
    try:
        combos = _parse_combos(args.combos)
    except Exception as exc:
        print(f"bad --combos: {exc}", file=sys.stderr)
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

    client = chroma_http_client(host=args.chroma_host, port=int(args.chroma_port))
    ef = build_chroma_embedding_function(sc)
    col = client.get_collection(str(sc["collection"]), embedding_function=ef)
    dense = DenseChromaRetriever(col, name="dense::full_bge_m3")

    encoder = FlagEmbeddingMultiVectorEncoder(
        model_name=args.bge_model,
        use_fp16=bool(args.bge_fp16),
        device=args.bge_device,
        batch_size=int(args.bge_batch_size),
    )

    write_eval_banner(
        "eval.retrieval.run_bge_m3_multi_vector_sweep",
        {
            "run_dir": str(run_root.relative_to(_ROOT)),
            "scenario": "full_bge_m3",
            "collection": sc["collection"],
            "candidate_top_k": args.candidate_top_k,
            "top_k": args.top_k,
            "combos": ",".join(c[0] for c in combos),
            "metric_modes": ",".join(metric_modes),
            "bge_model": args.bge_model,
            "bge_device": str(args.bge_device or "(default)"),
            "bge_batch_size": args.bge_batch_size,
            "eval_file": str(args.eval_file),
            "seed": args.seed,
            "gold_sha256": file_sha256(args.eval_file) or "(unavailable)",
            "git_sha": git_head_sha(_ROOT) or "(unavailable)",
        },
    )

    matrix_manifest = {
        "component": "eval.retrieval.run_bge_m3_multi_vector_sweep",
        "ts": ts,
        "scenario": "full_bge_m3",
        "collection_resolved": sc["collection"],
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "combos": [
            {"name": n, "w_dense": w[0], "w_sparse": w[1], "w_colbert": w[2]}
            for n, w in combos
        ],
        "metric_modes": metric_modes,
        "primary_metric_mode": primary_metric_mode,
        "bge_model": args.bge_model,
        "bge_device": args.bge_device,
        "bge_batch_size": args.bge_batch_size,
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

    summary: Dict[str, Any] = {
        "run_dir": str(run_root.relative_to(_ROOT)),
        "n_queries": len(dataset),
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
        "combos": [
            {"name": n, "w_dense": w[0], "w_sparse": w[1], "w_colbert": w[2]}
            for n, w in combos
        ],
        "results": {},
    }

    all_ok = True
    for name, (wd, ws, wc) in combos:
        retr = BgeM3MultiVectorRetriever(
            dense, encoder,
            candidate_top_k=int(args.candidate_top_k),
            w_dense=wd, w_sparse=ws, w_colbert=wc,
            name=f"bge_m3::{name}",
        )
        combo_dir = run_root / name
        combo_dir.mkdir(exist_ok=True)
        trace_path = combo_dir / f"{name}.trace.jsonl"
        json_path = combo_dir / f"{name}.json"
        try:
            with trace_path.open("w", encoding="utf-8") as tsink:
                out = evaluate_retriever(
                    retr,
                    dataset,
                    k_values=k_values,
                    fetch_k=int(args.top_k),
                    hit_fns=hit_fns,
                    primary_metric_mode=primary_metric_mode,
                    trace_sink=tsink,
                    trace_meta={
                        "combo": name,
                        "w_dense": wd,
                        "w_sparse": ws,
                        "w_colbert": wc,
                        "candidate_top_k": int(args.candidate_top_k),
                    },
                    progress_every=eval_progress_every_from_env(),
                    progress_prefix=f"F4 {name} ",
                )
            payload = {
                "_meta": {
                    "scenario": "full_bge_m3",
                    "combo": name,
                    "candidate_top_k": int(args.candidate_top_k),
                    "top_k": int(args.top_k),
                    "weights": {"dense": wd, "sparse": ws, "colbert": wc},
                    "metric_modes": metric_modes,
                    "primary_metric_mode": primary_metric_mode,
                    "n_queries": out["n_queries"],
                },
                **out["metrics"],
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["results"][name] = {
                "metrics": out["metrics"],
                "metrics_by_mode": out["metrics_by_mode"],
                "json": str(json_path.relative_to(_ROOT)),
                "trace": str(trace_path.relative_to(_ROOT)),
                "ok": True,
            }
        except Exception as exc:
            summary["results"][name] = {"ok": False, "error": str(exc)}
            print(f"FAIL combo={name}: {exc!r}", file=sys.stderr)
            all_ok = False

    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _write_metrics_markdown(run_root, summary)
    print(f"Done. Artifacts in {run_root}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Один вход для пост-обработки после retrieval с ``--retrieval-trace-jsonl``.

Запускает (по флагам) ``metrics_ladder_funnel`` и/или ``semantic_threshold_probe``,
как в `eval/METRICS_LADDER.md`, с единым контрактом путей.

Example::

    python -m eval.diagnostics.run_post_eval_ladder \\
        --trace eval/runs/<id>/full_bge_m3/retrieval_trace_baseline_mpnet_rerank.jsonl \\
        --corpus-jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl \\
        --ladder-out-json eval/runs/<id>/full_bge_m3/metrics_ladder_with_corpusA.json \\
        --probe-out-json eval/runs/<id>/full_bge_m3/semantic_probe.json \\
        --max-probe-rows 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[2]


def _run_module(mod: str, argv: List[str]) -> int:
    cmd = [sys.executable, "-m", mod, *argv]
    print("[run_post_eval_ladder]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True, help="retrieval_trace_*.jsonl")
    p.add_argument("--gold-jsonl", type=Path, default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl")
    p.add_argument("--corpus-jsonl", type=Path, default=None, help="Полный экспорт чанков для класса A (без --max-corpus-lines в финале).")
    p.add_argument("--ladder-out-json", type=Path, default=None, help="Если задан — запустить metrics_ladder_funnel.")
    p.add_argument("--ladder-hit-fn", choices=("substring", "token_f1"), default="substring")
    p.add_argument("--ladder-k-primary", type=int, default=10)
    p.add_argument("--probe-out-json", type=Path, default=None, help="Если задан — запустить semantic_threshold_probe.")
    p.add_argument("--max-probe-rows", type=int, default=30)
    args = p.parse_args()

    rc = 0
    if args.ladder_out_json is not None:
        ladder_argv: List[str] = [
            "--trace",
            str(args.trace.resolve()),
            "--gold-jsonl",
            str(args.gold_jsonl.resolve()),
            "--hit-fn",
            args.ladder_hit_fn,
            "--k-primary",
            str(int(args.ladder_k_primary)),
            "--out-json",
            str(args.ladder_out_json.resolve()),
        ]
        if args.corpus_jsonl is not None:
            ladder_argv.extend(["--corpus-jsonl", str(args.corpus_jsonl.resolve())])
        rc = _run_module("eval.diagnostics.metrics_ladder_funnel", ladder_argv)
        if rc != 0:
            return rc

    if args.probe_out_json is not None:
        probe_argv: List[str] = [
            "--trace",
            str(args.trace.resolve()),
            "--gold-jsonl",
            str(args.gold_jsonl.resolve()),
            "--out-json",
            str(args.probe_out_json.resolve()),
            "--max-rows",
            str(int(args.max_probe_rows)),
        ]
        prc = _run_module("eval.diagnostics.semantic_threshold_probe", probe_argv)
        rc = prc if rc == 0 else rc

    if args.ladder_out_json is None and args.probe_out_json is None:
        p.error("Укажите хотя бы один из --ladder-out-json или --probe-out-json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

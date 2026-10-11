"""Answer-eval for USER-bge-m3 supplement: 3 rerank profiles × 101 questions."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

USER_BGE_M3_CELLS: List[tuple[str, str]] = [
    ("full_user_bge_m3", "baseline_mpnet_rerank"),
    ("full_user_bge_m3", "rerank_bge_v2_m3"),
    ("full_user_bge_m3", "rerank_jina_v3"),
]

PRIMARY_METRIC = "substring::recall@10"


def _expected_rows(max_rows: int, eval_file: Path) -> int:
    if max_rows > 0:
        return max_rows
    if not eval_file.is_file():
        return 101
    n = sum(1 for line in eval_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return n if n > 0 else 101


def _cell_has_results(out_dir: Path, expected: int) -> bool:
    jsonl = out_dir / "answer_eval_results.jsonl"
    if not jsonl.is_file():
        return False
    n = sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
    return n >= expected


def _recall10_from_summary(summary: Dict[str, Any], sid: str, pname: str) -> float | None:
    cell = (summary.get("results") or {}).get(sid, {}).get(pname) or {}
    v = (cell.get("metrics") or {}).get(PRIMARY_METRIC)
    return float(v) if isinstance(v, (int, float)) else None


def _ensure_gpu_env() -> None:
    os.environ.setdefault("GRAPH_RAG_ENABLED", "0")
    os.environ.setdefault("LLM_CACHE_ENABLED", "0")
    os.environ.setdefault("ST_EMBED_DEVICE", "cuda")
    os.environ.setdefault("RERANK_DEVICE", "cuda")
    os.environ.setdefault("ST_EMBED_BATCH_SIZE", "32")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b3-run-dir", type=Path, required=True, help="Supplemental USER-bge-m3 B3 run dir")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_answer_user_bge_m3.json",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    _ensure_gpu_env()

    run_dir = args.b3_run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        print(f"missing {summary_path}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_file = _ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl"
    expected = _expected_rows(int(args.max_rows), eval_file)

    candidates: List[Dict[str, Any]] = []
    for sid, pname in USER_BGE_M3_CELLS:
        retrieval_json = run_dir / sid / f"retrieval_{pname}.json"
        candidates.append(
            {
                "scenario": sid,
                "profile": pname,
                "retrieval_json": str(retrieval_json),
                "exists": retrieval_json.is_file(),
            }
        )

    if args.dry_run:
        print(json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2))
        return 0 if all(c.get("exists") for c in candidates) else 2

    results: List[Dict[str, Any]] = []
    for rank, (sid, pname) in enumerate(USER_BGE_M3_CELLS, 1):
        retrieval_json = run_dir / sid / f"retrieval_{pname}.json"
        if not retrieval_json.is_file():
            print(f"skip missing {retrieval_json}", flush=True)
            continue
        out_dir = run_dir / "_answer_user_bge_m3" / f"{rank:02d}_{sid}_{pname}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and _cell_has_results(out_dir, expected):
            print(f"skip existing results: {out_dir}", flush=True)
            manifest = out_dir / "answer_eval_manifest.json"
            row: Dict[str, Any] = {
                "rank": rank,
                "substring_recall_at_10": _recall10_from_summary(summary, sid, pname),
                "scenario": sid,
                "profile": pname,
                "retrieval_json": str(retrieval_json.relative_to(_ROOT)),
                "returncode": 0,
                "skipped": True,
                "manifest": str(manifest.relative_to(_ROOT)) if manifest.is_file() else None,
            }
            if manifest.is_file():
                row["manifest_data"] = json.loads(manifest.read_text(encoding="utf-8"))
            results.append(row)
            continue
        cmd = [
            sys.executable,
            "-m",
            "eval.diagnostics.run_answer_eval_in_run_dir",
            "--run-dir",
            str(out_dir),
            "--retrieval-json",
            str(retrieval_json),
            "--max-rows",
            str(int(args.max_rows)),
        ]
        print("[answer_user_bge_m3]", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        (out_dir / "answer_eval.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "answer_eval.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        manifest = out_dir / "answer_eval_manifest.json"
        row = {
            "rank": rank,
            "substring_recall_at_10": _recall10_from_summary(summary, sid, pname),
            "scenario": sid,
            "profile": pname,
            "retrieval_json": str(retrieval_json.relative_to(_ROOT)),
            "returncode": proc.returncode,
            "manifest": str(manifest.relative_to(_ROOT)) if manifest.is_file() else None,
        }
        if manifest.is_file():
            row["manifest_data"] = json.loads(manifest.read_text(encoding="utf-8"))
        results.append(row)

    payload = {
        "b3_run_dir": str(run_dir.relative_to(_ROOT)),
        "metric": PRIMARY_METRIC,
        "cells": results,
        "candidates_planned": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(r.get("returncode") == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

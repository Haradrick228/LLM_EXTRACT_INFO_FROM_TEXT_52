"""Answer-eval on diverse B3 candidates (by metric leaders, not only substring top-3)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.reporting.decision_analysis import DECISION_METRIC_KEYS, _rank_metric

# Explicit candidates: (scenario, profile, label)
DEFAULT_CANDIDATES: List[Tuple[str, str, str]] = [
    ("full_frida", "rerank_jina_v3", "canonical_protocol"),
    ("full_frida", "rerank_bge_v2_m3", "substring_tie_best_ref_sim"),
    ("full_bge_m3", "rerank_bge_v2_m3", "token_f1_and_semantic_mrr_leader"),
    ("full_jina_embeddings_v3", "rerank_bge_v2_m3", "semantic_recall_leader"),
]


def _leaders_from_summary(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for mk in DECISION_METRIC_KEYS:
        ranked = _rank_metric(summary, mk)
        if not ranked:
            continue
        best_v = ranked[0][0]
        for v, sid, pname in ranked:
            if abs(v - best_v) > 1e-9:
                break
            key = (sid, pname)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b3-run-dir", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_answer_diverse.json",
    )
    p.add_argument(
        "--skip-jina-embedder",
        action="store_true",
        help="Skip full_jina_embeddings_v3 (unstable remote code in some envs).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write candidate list, do not run answer eval.",
    )
    args = p.parse_args()

    run_dir = args.b3_run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        print(f"missing {summary_path}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seen: Set[Tuple[str, str]] = set()
    candidates: List[Dict[str, str]] = []
    for sid, pname, tag in DEFAULT_CANDIDATES:
        if args.skip_jina_embedder and sid == "full_jina_embeddings_v3":
            continue
        key = (sid, pname)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"scenario": sid, "profile": pname, "reason": tag})

    for sid, pname in _leaders_from_summary(summary):
        if args.skip_jina_embedder and sid == "full_jina_embeddings_v3":
            continue
        key = (sid, pname)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"scenario": sid, "profile": pname, "reason": "metric_leader"})

    if args.dry_run:
        print(json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2))
        return 0

    results: List[Dict[str, Any]] = []
    rank = 0
    for cand in candidates:
        sid = cand["scenario"]
        pname = cand["profile"]
        scen_dir = run_dir / sid
        retrieval_json = scen_dir / f"retrieval_{pname}.json"
        if not retrieval_json.is_file():
            print(f"skip missing {retrieval_json}", flush=True)
            continue
        rank += 1
        out_dir = run_dir / "_answer_diverse" / f"{rank:02d}_{sid}_{pname}"
        out_dir.mkdir(parents=True, exist_ok=True)
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
        print("[answer_diverse]", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        (out_dir / "answer_eval.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "answer_eval.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        manifest = out_dir / "answer_eval_manifest.json"
        row: Dict[str, Any] = {
            "rank": rank,
            "scenario": sid,
            "profile": pname,
            "reason": cand.get("reason", ""),
            "retrieval_json": str(retrieval_json.relative_to(_ROOT)),
            "returncode": proc.returncode,
            "manifest": str(manifest.relative_to(_ROOT)) if manifest.is_file() else None,
        }
        if manifest.is_file():
            row["manifest_data"] = json.loads(manifest.read_text(encoding="utf-8"))
        results.append(row)

    payload = {
        "b3_run_dir": str(run_dir.relative_to(_ROOT)),
        "cells": results,
        "candidates_planned": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(r.get("returncode") == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

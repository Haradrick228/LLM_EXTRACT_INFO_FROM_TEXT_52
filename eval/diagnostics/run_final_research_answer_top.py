"""Answer-eval on top-N B3 cells by substring::recall@10."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _top_cells(summary: Dict[str, Any], metric: str) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    for sid, profs in (summary.get("results") or {}).items():
        for pname, cell in (profs or {}).items():
            m = cell.get("metrics") or {}
            v = m.get(metric)
            if isinstance(v, (int, float)):
                rows.append((float(v), str(sid), str(pname)))
    rows.sort(reverse=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b3-run-dir", type=Path, required=True)
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--metric", default="substring::recall@10")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_answer_top.json",
    )
    args = p.parse_args()

    run_dir = args.b3_run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        print(f"missing {summary_path}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ranked = _top_cells(summary, args.metric)
    results: List[Dict[str, Any]] = []

    rank = 0
    for score, sid, pname in ranked:
        if rank >= int(args.top_n):
            break
        # Jina embedder in this environment can require extra remote code fetches;
        # keep the pipeline stable by skipping these candidates in top answer-eval.
        if sid == "full_jina_embeddings_v3":
            continue
        scen_dir = run_dir / sid
        retrieval_json = scen_dir / f"retrieval_{pname}.json"
        candidate_rank = rank + 1
        out_dir = run_dir / "_answer_top" / f"{candidate_rank:02d}_{sid}_{pname}"
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
        print("[answer_top]", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        (out_dir / "answer_eval.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "answer_eval.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        manifest = out_dir / "answer_eval_manifest.json"
        row: Dict[str, Any] = {
            "rank": candidate_rank,
            "substring_recall_at_10": score,
            "scenario": sid,
            "profile": pname,
            "retrieval_json": str(retrieval_json.relative_to(_ROOT)),
            "returncode": proc.returncode,
            "manifest": str(manifest.relative_to(_ROOT)) if manifest.is_file() else None,
        }
        if manifest.is_file():
            row["manifest_data"] = json.loads(manifest.read_text(encoding="utf-8"))
        if proc.returncode == 0:
            rank += 1
            results.append(row)

    payload = {"b3_run_dir": str(run_dir.relative_to(_ROOT)), "metric": args.metric, "cells": results}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

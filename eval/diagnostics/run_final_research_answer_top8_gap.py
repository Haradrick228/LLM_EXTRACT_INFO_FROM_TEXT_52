"""Answer-eval for top-8 gap cells (5 missing avg_ref_sim), same pipeline as run_final_research_answer_top."""
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

# Rows 3–7 in outcomes «Топ комбинаций» (FRIDA×3 already in final_research_answer_top.json).
TOP8_GAP_CANDIDATES: List[Tuple[str, str, str]] = [
    ("full_jina_embeddings_v3", "rerank_bge_v2_m3", "top8_row3"),
    ("full_e5_large_instruct", "rerank_jina_v3", "top8_row4"),
    ("full_e5_large_instruct", "rerank_bge_v2_m3", "top8_row5"),
    ("full_bge_m3", "rerank_bge_v2_m3", "top8_row6"),
    ("full_bge_m3", "rerank_jina_v3", "top8_row7"),
]

PRIMARY_METRIC = "substring::recall@10"


def _existing_ok_keys(top_json: Path) -> Set[Tuple[str, str]]:
    if not top_json.is_file():
        return set()
    payload = json.loads(top_json.read_text(encoding="utf-8"))
    out: Set[Tuple[str, str]] = set()
    for cell in payload.get("cells") or []:
        if cell.get("returncode") != 0:
            continue
        out.add((str(cell.get("scenario") or ""), str(cell.get("profile") or "")))
    return out


def _recall10_from_summary(summary: Dict[str, Any], sid: str, pname: str) -> float | None:
    cell = (summary.get("results") or {}).get(sid, {}).get(pname) or {}
    v = (cell.get("metrics") or {}).get(PRIMARY_METRIC)
    return float(v) if isinstance(v, (int, float)) else None


def _parse_only_filters(raw: List[str]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for item in raw:
        if ":" not in item:
            raise SystemExit(f"--only must be scenario:profile, got {item!r}")
        sid, pname = item.split(":", 1)
        out.add((sid.strip(), pname.strip()))
    return out


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b3-run-dir", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_answer_top8_gap.json",
    )
    p.add_argument(
        "--answer-top-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_answer_top.json",
        help="Skip (scenario, profile) already OK here (default: existing FRIDA×3).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list candidates and retrieval paths.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cell if answer_eval_results.jsonl already has enough rows.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SCENARIO:PROFILE",
        help="Run only these gap cells (repeatable).",
    )
    args = p.parse_args()

    run_dir = args.b3_run_dir.resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        print(f"missing {summary_path}", file=sys.stderr)
        return 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    skip_keys = _existing_ok_keys(args.answer_top_json.resolve())
    only_keys = _parse_only_filters(args.only) if args.only else None
    eval_file = _ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl"
    expected = _expected_rows(int(args.max_rows), eval_file)

    candidates: List[Dict[str, Any]] = []
    for sid, pname, tag in TOP8_GAP_CANDIDATES:
        key = (sid, pname)
        if only_keys is not None and key not in only_keys:
            continue
        if key in skip_keys:
            print(f"skip existing OK: {sid} + {pname}", flush=True)
            continue
        retrieval_json = run_dir / sid / f"retrieval_{pname}.json"
        candidates.append(
            {
                "scenario": sid,
                "profile": pname,
                "reason": tag,
                "retrieval_json": str(retrieval_json),
                "exists": retrieval_json.is_file(),
            }
        )

    if args.dry_run:
        print(json.dumps({"candidates": candidates, "skip_keys": list(skip_keys)}, ensure_ascii=False, indent=2))
        return 0 if all(c.get("exists") for c in candidates) else 2

    results: List[Dict[str, Any]] = []
    rank = 0
    for cand in candidates:
        if not cand.get("exists"):
            print(f"skip missing {cand['retrieval_json']}", flush=True)
            continue
        sid = str(cand["scenario"])
        pname = str(cand["profile"])
        retrieval_json = Path(cand["retrieval_json"])
        rank += 1
        out_dir = run_dir / "_answer_top8_gap" / f"{rank:02d}_{sid}_{pname}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and _cell_has_results(out_dir, expected):
            print(f"skip existing results: {out_dir}", flush=True)
            manifest = out_dir / "answer_eval_manifest.json"
            row = {
                "rank": rank,
                "substring_recall_at_10": _recall10_from_summary(summary, sid, pname),
                "scenario": sid,
                "profile": pname,
                "reason": cand.get("reason", ""),
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
        print("[answer_top8_gap]", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        (out_dir / "answer_eval.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (out_dir / "answer_eval.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        manifest = out_dir / "answer_eval_manifest.json"
        row: Dict[str, Any] = {
            "rank": rank,
            "substring_recall_at_10": _recall10_from_summary(summary, sid, pname),
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

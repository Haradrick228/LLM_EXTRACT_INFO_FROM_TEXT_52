"""
Run retrieval (and optionally answer) eval for each rerank profile in rag_profiles.yaml (matrix: true).

Writes under eval/runs/<timestamp>_rerank_matrix/ without deleting prior runs.
Restores os.environ after each profile.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import apply_profile, list_profile_names  # noqa: E402


def _redacted_env() -> Dict[str, str]:
    skip = ("KEY", "SECRET", "PASSWORD", "TOKEN", "DSN")
    out: Dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        if any(s in k.upper() for s in skip):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval eval for each matrix rerank profile.")
    parser.add_argument(
        "--profiles",
        default="matrix",
        help="Comma-separated profile names, or 'matrix' for all profiles with matrix: true",
    )
    parser.add_argument("--eval-file", default=str(_ROOT / "eval" / "eval_set_gold_v2.jsonl"))
    parser.add_argument("--embed-model", default=os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    parser.add_argument("--fetch-k", type=int, default=int(os.getenv("RETRIEVER_FETCH_K", "40") or 40))
    parser.add_argument("--k", default="5,10")
    parser.add_argument("--skip-answer", action="store_true", help="Do not run run_answer_eval.py")
    parser.add_argument(
        "--retrieval-max-rows",
        type=int,
        default=int(os.getenv("RETRIEVAL_EVAL_MAX_ROWS", "0") or 0),
        help="Передать в run_retrieval_eval --max-rows (0 = все вопросы в eval-файле).",
    )
    args = parser.parse_args()

    if args.profiles.strip() == "matrix":
        names: List[str] = list_profile_names(matrix_only=True)
    else:
        names = [x.strip() for x in args.profiles.split(",") if x.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _ROOT / "eval" / "runs" / f"{ts}_rerank_matrix"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline = dict(os.environ)
    summary: Dict[str, object] = {"run_dir": str(run_dir.relative_to(_ROOT)), "profiles": names, "results": {}}

    eval_chroma_host = baseline.get("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    eval_chroma_port = baseline.get("RAG_EVAL_CHROMA_PORT", "18000")

    for name in names:
        os.environ.clear()
        os.environ.update(baseline)
        applied = apply_profile(name)
        (run_dir / f"profile_{name}_env_applied.json").write_text(
            json.dumps(applied, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        out_json = run_dir / f"retrieval_{name}.json"
        cmd = [
            sys.executable,
            "-m",
            "eval.run_retrieval_eval",
            "--eval-file",
            args.eval_file,
            "--embed-model",
            args.embed_model,
            "--fetch-k",
            str(args.fetch_k),
            "--k",
            args.k,
            "--rerank-model",
            os.getenv("RERANK_MODEL", ""),
            "--rerank-mode",
            os.getenv("RERANK_MODE", "auto"),
            "--chroma-host",
            eval_chroma_host,
            "--chroma-port",
            str(eval_chroma_port),
            "--out-json",
            str(out_json),
        ]
        if args.retrieval_max_rows > 0:
            cmd.extend(["--max-rows", str(args.retrieval_max_rows)])
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        (run_dir / f"retrieval_{name}.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / f"retrieval_{name}.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        summary["results"][name] = {"retrieval_returncode": proc.returncode}

        if not args.skip_answer and proc.returncode == 0 and out_json.is_file():
            out_ans = run_dir / f"answer_{name}.jsonl"
            cmd_a = [
                sys.executable,
                "-m",
                "eval.run_answer_eval",
            ]
            env_a = dict(os.environ)
            env_a["ANSWER_EVAL_FILE"] = args.eval_file
            env_a["ANSWER_EVAL_OUT"] = str(out_ans)
            env_a["ANSWER_EVAL_FETCH_K"] = str(args.fetch_k)
            proc_a = subprocess.run(cmd_a, cwd=str(_ROOT), capture_output=True, text=True, env=env_a)
            (run_dir / f"answer_{name}.stdout.txt").write_text(proc_a.stdout or "", encoding="utf-8")
            (run_dir / f"answer_{name}.stderr.txt").write_text(proc_a.stderr or "", encoding="utf-8")
            summary["results"][name]["answer_returncode"] = proc_a.returncode

    os.environ.clear()
    os.environ.update(baseline)

    (run_dir / "env_snapshot.json").write_text(
        json.dumps(_redacted_env(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Artifacts in", run_dir)


if __name__ == "__main__":
    main()

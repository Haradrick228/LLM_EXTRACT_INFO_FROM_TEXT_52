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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import apply_profile, list_profile_names  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402


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
    parser.add_argument(
        "--eval-file",
        default=str(_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl"),
    )
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
    parser.add_argument(
        "--chroma-collection",
        default="",
        help="Имя коллекции Chroma для всех прогонов (пусто = из окружения CHROMA_COLLECTION). Передаётся в subprocess.",
    )
    parser.add_argument(
        "--embed-trust-remote-code",
        action="store_true",
        help="Передать в run_retrieval_eval --embed-trust-remote-code (Jina и др.).",
    )
    parser.add_argument(
        "--metric-modes",
        default=(os.getenv("RETRIEVAL_METRIC_MODES", "substring,token_f1,semantic") or "substring,token_f1,semantic").strip(),
        help="Передать в run_retrieval_eval --metric-modes.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.62") or 0.62),
    )
    parser.add_argument(
        "--semantic-model",
        default=os.getenv(
            "RETRIEVAL_SEMANTIC_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ).strip(),
    )
    args = parser.parse_args()

    if args.profiles.strip() == "matrix":
        names: List[str] = list_profile_names(matrix_only=True)
    else:
        names = [x.strip() for x in args.profiles.split(",") if x.strip()]

    baseline = dict(os.environ)
    if args.chroma_collection.strip():
        baseline["CHROMA_COLLECTION"] = args.chroma_collection.strip()
    baseline.setdefault("RETRIEVAL_EVAL_SHUFFLE", "0")
    baseline.setdefault("EVAL_SCENARIO_TYPE", os.getenv("EVAL_SCENARIO_TYPE", "fixture_smoke"))

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _ROOT / "eval" / "runs" / f"{ts}_rerank_matrix"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "run_dir": str(run_dir.relative_to(_ROOT)),
        "profiles": names,
        "eval_file": args.eval_file,
        "embed_model": args.embed_model,
        "fetch_k": args.fetch_k,
        "k": args.k,
        "metric_modes": args.metric_modes,
        "semantic_threshold": args.semantic_threshold,
        "semantic_model": args.semantic_model,
        "eval_scenario_type": baseline.get("EVAL_SCENARIO_TYPE", ""),
        "retrieval_eval_shuffle": baseline.get("RETRIEVAL_EVAL_SHUFFLE", "0"),
        "embed_trust_remote_code": args.embed_trust_remote_code,
        "results": {},
    }
    chunking_hint = baseline.get("CHUNKING_LABEL") or baseline.get("CHUNK_PROFILE") or ""
    chroma_col = baseline.get("CHROMA_COLLECTION", "default")
    write_eval_banner(
        "run_profile_metrics",
        {
            "eval_file": args.eval_file,
            "embed_model": args.embed_model,
            "fetch_k": args.fetch_k,
            "k": args.k,
            "chroma_collection": chroma_col,
            "matrix_profiles": ", ".join(names),
            "retrieval_max_rows": args.retrieval_max_rows,
            "chunking_label": chunking_hint or "(not set)",
        },
    )

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
        write_eval_banner(
            f"run_profile_metrics[profile={name}]",
            {
                "rag_profile": name,
                "rerank_model": os.getenv("RERANK_MODEL", ""),
                "rerank_mode": os.getenv("RERANK_MODE", ""),
                "rerank_device": os.getenv("RERANK_DEVICE", ""),
                "embed_model": args.embed_model,
                "chroma_collection": os.getenv("CHROMA_COLLECTION", "default"),
                "eval_file": args.eval_file,
                "fetch_k": args.fetch_k,
            },
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
            "--metric-modes",
            args.metric_modes,
            "--semantic-threshold",
            str(args.semantic_threshold),
            "--semantic-model",
            args.semantic_model,
            "--rerank-model",
            os.getenv("RERANK_MODEL", ""),
            "--rerank-mode",
            os.getenv("RERANK_MODE", "auto"),
            "--chroma-host",
            eval_chroma_host,
            "--chroma-port",
            str(eval_chroma_port),
            "--chroma-collection",
            os.getenv("CHROMA_COLLECTION", "default"),
            "--out-json",
            str(out_json),
        ]
        if args.retrieval_max_rows > 0:
            cmd.extend(["--max-rows", str(args.retrieval_max_rows)])
        if args.embed_trust_remote_code:
            cmd.append("--embed-trust-remote-code")
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

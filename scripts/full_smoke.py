"""
Полный смоук без фазы B: стек (Chroma/Ollama/CUDA) → retrieval → опционально answer-eval (Ollama).

  python scripts/full_smoke.py
  python scripts/full_smoke.py --with-llm

Фаза B не затрагивается (нет новых коллекций / triple-dense).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import load_profiles  # noqa: E402


def _default_embed_model() -> str:
    local = _ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    if (local / "config.json").is_file():
        return str(local)
    return os.getenv(
        "CHROMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )


def _profile_env(name: str) -> Dict[str, str]:
    data = load_profiles()
    prof = data["profiles"][name]
    out: Dict[str, str] = {}
    skip = frozenset({"description", "matrix", "version"})
    for k, v in prof.items():
        if k in skip or v is None:
            continue
        out[str(k)] = str(v)
    return out


def _run(cmd: list[str], *, env: dict[str, str], label: str) -> int:
    print(f"\n[full_smoke] === {label} ===", flush=True)
    print("[full_smoke]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if r.returncode != 0:
        print(f"[full_smoke] FAIL {label} exit={r.returncode}", flush=True)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Full RAG smoke: verify stack + retrieval (+ optional LLM).")
    p.add_argument(
        "--with-llm",
        action="store_true",
        help="После retrieval вызвать run_answer_eval (2 вопроса, Ollama).",
    )
    p.add_argument(
        "--retrieval-rows",
        type=int,
        default=10,
        help="Сколько строк из sample20 для retrieval (по умолчанию 10).",
    )
    p.add_argument(
        "--answer-rows",
        type=int,
        default=2,
        help="При --with-llm: сколько вопросов для answer-eval.",
    )
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "eval" / "runs" / f"{ts}_full_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = os.environ.copy()
    base.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    base.setdefault("RAG_EVAL_CHROMA_PORT", "18000")
    base.setdefault("CHROMA_COLLECTION", os.getenv("CHROMA_COLLECTION", "default"))
    embed = _default_embed_model()
    base["CHROMA_EMBED_MODEL"] = embed
    base.setdefault("RERANK_DEVICE", os.getenv("RERANK_DEVICE", "cuda"))

    if _run(
        [sys.executable, str(_ROOT / "scripts" / "verify_local_stack.py")],
        env=base,
        label="verify_local_stack",
    ):
        return 1

    retr_env = dict(base)
    retr_env.update(_profile_env("baseline_mpnet_rerank"))
    retr_json = out_dir / "retrieval_sample20.json"
    if _run(
        [
            sys.executable,
            "-m",
            "eval.run_retrieval_eval",
            "--eval-file",
            str(_ROOT / "eval" / "eval_set_with_answer_ref_sample20.jsonl"),
            "--embed-model",
            embed,
            "--fetch-k",
            "40",
            "--k",
            "5,10",
            "--rerank-model",
            retr_env.get("RERANK_MODEL", ""),
            "--rerank-mode",
            retr_env.get("RERANK_MODE", "auto"),
            "--chroma-host",
            retr_env.get("RAG_EVAL_CHROMA_HOST", "127.0.0.1"),
            "--chroma-port",
            str(retr_env.get("RAG_EVAL_CHROMA_PORT", "18000")),
            "--max-rows",
            str(args.retrieval_rows),
            "--out-json",
            str(retr_json),
        ],
        env=retr_env,
        label="run_retrieval_eval (sample20)",
    ):
        return 2

    if not args.with_llm:
        print("\n[full_smoke] OK (без LLM). Артефакты:", out_dir, flush=True)
        return 0

    ans_env = dict(base)
    ans_env.update(_profile_env("ollama_answer_eval"))
    ans_env["CHROMA_EMBED_MODEL"] = embed
    ans_env["ANSWER_EVAL_FILE"] = str(_ROOT / "eval" / "eval_set_with_answer_ref_sample20.jsonl")
    ans_env["ANSWER_EVAL_OUT"] = str(out_dir / "answer_sample.jsonl")
    ans_env["ANSWER_EVAL_MAX_ROWS"] = str(max(1, args.answer_rows))
    ans_env["ANSWER_EVAL_FETCH_K"] = "40"

    if _run([sys.executable, "-m", "eval.run_answer_eval"], env=ans_env, label="run_answer_eval (Ollama)"):
        return 3

    print("\n[full_smoke] OK (с LLM). Артефакты:", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

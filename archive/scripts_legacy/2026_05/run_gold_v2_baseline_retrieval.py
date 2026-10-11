"""
Прогон retrieval-eval на golden v2 с первым профилем матрицы: baseline_mpnet_rerank.

  python scripts/run_gold_v2_baseline_retrieval.py
  python scripts/run_gold_v2_baseline_retrieval.py --out-json eval/runs/my_run.json

По умолчанию коллекция: `CHROMA_COLLECTION` из окружения или **`default`**. После успешного
`sync_gold_v2_fixture_collection.py` выставьте `CHROMA_COLLECTION=gold_v2_fixture` для совпадения с фикстурой 1:1.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

PROFILE = "baseline_mpnet_rerank"


def _embed_model() -> str:
    local = _ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    if (local / "config.json").is_file():
        return str(local)
    return os.getenv(
        "CHROMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "gold_v2_baseline_mpnet_rerank.json",
    )
    p.add_argument("--fetch-k", type=int, default=40)
    args = p.parse_args()

    sys.path.insert(0, str(_ROOT))
    from common.rag_profile_env import apply_profile  # noqa: E402

    os.environ.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    os.environ.setdefault("RAG_EVAL_CHROMA_PORT", "18000")
    os.environ.setdefault("CHROMA_COLLECTION", os.getenv("CHROMA_COLLECTION", "default"))
    os.environ.setdefault("RERANK_DEVICE", os.getenv("RERANK_DEVICE", "cuda"))

    apply_profile(PROFILE)
    embed = _embed_model()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "eval.run_retrieval_eval",
        "--eval-file",
        str(_ROOT / "eval" / "eval_set_gold_v2.jsonl"),
        "--embed-model",
        embed,
        "--fetch-k",
        str(args.fetch_k),
        "--k",
        "5,10",
        "--rerank-model",
        os.getenv("RERANK_MODEL", ""),
        "--rerank-mode",
        os.getenv("RERANK_MODE", "cross"),
        "--out-json",
        str(args.out_json),
    ]
    print("[run_gold_v2]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT), env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())

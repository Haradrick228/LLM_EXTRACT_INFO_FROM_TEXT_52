"""
Смоук фазы A: Chroma + один профиль реранкера + gold_v2 (без полной матрицы).

  python scripts/smoke_phase_a.py

Ожидается: Docker Chroma на localhost (см. RAG_EVAL_CHROMA_*), веса HF доступны или локальный CHROMA_EMBED_MODEL.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _default_embed_model() -> str:
    local = _ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    if (local / "config.json").is_file():
        return str(local)
    return os.getenv(
        "CHROMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )


def main() -> int:
    os.environ.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    os.environ.setdefault("RAG_EVAL_CHROMA_PORT", "18000")
    os.environ.setdefault("CHROMA_COLLECTION", os.getenv("CHROMA_COLLECTION", "default"))

    embed = _default_embed_model()
    os.environ["CHROMA_EMBED_MODEL"] = embed

    cmd = [
        sys.executable,
        "-m",
        "eval.run_profile_metrics",
        "--profiles",
        "baseline_mpnet_rerank",
        "--eval-file",
        str(_ROOT / "eval" / "eval_set_gold_v2.jsonl"),
        "--embed-model",
        embed,
        "--fetch-k",
        "20",
        "--k",
        "5,10",
        "--skip-answer",
    ]
    print("[smoke_phase_a]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

"""Интеграция с поднятым Chroma (Docker). По умолчанию пропускается."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _chroma_integration_enabled() -> bool:
    return os.getenv("CHROMA_INTEGRATION", "").strip().lower() in ("1", "true", "yes")


@pytest.mark.integration
@pytest.mark.skipif(not _chroma_integration_enabled(), reason="Set CHROMA_INTEGRATION=1 to run")
def test_chroma_v2_heartbeat():
    from urllib.request import urlopen

    host = os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1"))
    port = int(os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000")))
    with urlopen(f"http://{host}:{port}/api/v2/heartbeat", timeout=10) as r:
        body = r.read().decode("utf-8")
    assert r.status == 200
    assert "heartbeat" in body


@pytest.mark.integration
@pytest.mark.skipif(not _chroma_integration_enabled(), reason="Set CHROMA_INTEGRATION=1 to run")
def test_run_retrieval_eval_one_row_subprocess():
    env = os.environ.copy()
    env.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    env.setdefault("RAG_EVAL_CHROMA_PORT", "18000")
    env.setdefault("CHROMA_COLLECTION", "default")
    env.setdefault("RERANK_DEVICE", "cuda")
    env.update(
        {
            "RERANK_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "RERANK_MODE": "cross",
        }
    )
    local_mpnet = ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    embed = (
        str(local_mpnet)
        if (local_mpnet / "config.json").is_file()
        else os.getenv(
            "CHROMA_EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
    )
    cmd = [
        sys.executable,
        "-m",
        "eval.run_retrieval_eval",
        "--eval-file",
        str(ROOT / "eval" / "eval_set_with_answer_ref_sample20.jsonl"),
        "--embed-model",
        embed,
        "--fetch-k",
        "10",
        "--k",
        "5",
        "--max-rows",
        "1",
        "--rerank-model",
        env["RERANK_MODEL"],
        "--rerank-mode",
        env["RERANK_MODE"],
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr + proc.stdout

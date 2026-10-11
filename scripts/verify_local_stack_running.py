#!/usr/bin/env python3
"""Quick checks: Chroma HTTP, Ollama, CUDA — без загрузки HF-моделей."""
from __future__ import annotations

import json
import os
import sys
import urllib.request


def _get(url: str, timeout: float = 8.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def main() -> int:
    chroma_host = os.getenv("CHROMA_VERIFY_HOST", "127.0.0.1")
    chroma_port = os.getenv("CHROMA_VERIFY_PORT", "18000")
    ollama_host = os.getenv("OLLAMA_VERIFY_HOST", "127.0.0.1")
    ollama_port = os.getenv("OLLAMA_VERIFY_PORT", "11434")

    ok = True
    # Chroma
    try:
        code, body = _get(f"http://{chroma_host}:{chroma_port}/api/v2/heartbeat")
        print(f"[Chroma] heartbeat HTTP {code}: {body[:120]}")
    except Exception as e:
        ok = False
        print("[Chroma] FAIL:", e)

    try:
        code, body = _get(
            f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections"
        )
        cols = json.loads(body)
        names = [c.get("name") for c in cols if isinstance(c, dict)]
        tail = "..." if len(names) > 12 else ""
        print(f"[Chroma] collections ({len(names)}):", ", ".join(names[:12]) + tail)
    except Exception as e:
        ok = False
        print("[Chroma] list collections FAIL:", e)

    # Ollama
    try:
        code, body = _get(f"http://{ollama_host}:{ollama_port}/api/tags")
        data = json.loads(body)
        models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        print(f"[Ollama] tags HTTP {code}; models:", ", ".join(models[:8]) or "(none)")
    except Exception as e:
        ok = False
        print("[Ollama] FAIL:", e)

    # CUDA (optional torch)
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else ""
        print(f"[PyTorch] cuda_available={cuda}", f"device={name!r}" if cuda else "")
    except Exception as e:
        print("[PyTorch] skip or FAIL:", e)

    print()
    print("Подсказка: при запуске ml_service с хоста Windows переопределите Chroma:")
    print("  CHROMA_HOST=127.0.0.1 CHROMA_PORT=18000")
    print("  (в .env для Docker часто chroma:8000 — это только внутри compose-сети)")
    print("Реранкер/эмбеддеры: RERANK_DEVICE=cuda; при сбоях загрузки с HuggingFace проверьте VPN/кэш HF_HOME.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

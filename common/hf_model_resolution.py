from __future__ import annotations

import os
from pathlib import Path


def _hub_root() -> Path:
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir_name(repo_id: str) -> str:
    org, name = repo_id.split("/", 1)
    return f"models--{org}--{name}"


def _snapshot_from_refs(repo_dir: Path) -> Path | None:
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        rev = ref_main.read_text(encoding="utf-8").strip()
        cand = repo_dir / "snapshots" / rev
        if cand.is_dir():
            return cand
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    cands = [p for p in snaps.iterdir() if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_hf_model_path(model_name_or_path: str) -> str:
    """Return local snapshot path if present, otherwise keep original model id.

    This avoids spurious network calls (`model_info`) in offline/unstable network
    environments when a model is already present in HF cache.
    """
    raw = (model_name_or_path or "").strip()
    if not raw:
        return raw

    p = Path(raw)
    if p.exists():
        return str(p.resolve())

    # Not a HF repo id, leave as-is.
    if "/" not in raw or raw.startswith(("http://", "https://")):
        return raw

    repo_dir = _hub_root() / _repo_dir_name(raw)
    if not repo_dir.is_dir():
        return raw
    snap = _snapshot_from_refs(repo_dir)
    if snap is None:
        return raw
    return str(snap.resolve())


def embedder_needs_trust_remote_code(model_name_or_path: str) -> bool:
    """HF embedders that require trust_remote_code (Jina v3, Giga, etc.)."""
    raw = (model_name_or_path or "").strip().lower()
    if not raw:
        return False
    return any(
        token in raw
        for token in (
            "jina-embeddings-v3",
            "jina_embeddings_v3",
            "giga-embeddings",
            "giga_embeddings",
        )
    )


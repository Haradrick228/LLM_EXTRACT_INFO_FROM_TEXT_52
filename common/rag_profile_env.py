"""RAG experiment profiles: YAML in eval/rag_profiles.yaml -> os.environ."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_META_KEYS = frozenset({"description", "matrix", "version"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profiles_path() -> Path:
    return repo_root() / "eval" / "rag_profiles.yaml"


def load_profiles() -> Dict[str, Any]:
    path = profiles_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing profiles file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "profiles" not in data or not isinstance(data["profiles"], dict):
        raise ValueError("rag_profiles.yaml must contain a 'profiles' mapping")
    return data


def list_profile_names(*, matrix_only: bool = False) -> List[str]:
    data = load_profiles()
    out: List[str] = []
    for name, body in data["profiles"].items():
        if not isinstance(body, dict):
            continue
        if matrix_only and not body.get("matrix"):
            continue
        out.append(str(name))
    return sorted(out)


def apply_profile(name: str) -> Dict[str, str]:
    data = load_profiles()
    prof = data["profiles"].get(name)
    if not isinstance(prof, dict):
        raise KeyError(f"Unknown profile: {name!r}")

    applied: Dict[str, str] = {}
    for key, val in prof.items():
        if key in _META_KEYS:
            continue
        if val is None:
            continue
        os.environ[str(key)] = str(val)
        applied[str(key)] = str(val)
    return applied


def apply_profile_by_rag_profile_env() -> Optional[str]:
    raw = (os.getenv("RAG_PROFILE") or "").strip()
    if not raw:
        return None
    apply_profile(raw)
    return raw

"""Загрузка и валидация ``embedding_bases.yaml``."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_VAR_DEFAULT = re.compile(r"\$\{([^}:]+):-([^}]*)\}")


def expand_str(s: str, extra: Optional[Mapping[str, str]] = None) -> str:
    extra = extra or {}

    def repl(m: re.Match[str]) -> str:
        var, dfl = m.group(1), m.group(2)
        v = os.environ.get(var)
        if v is None or v == "":
            v = extra.get(var)
        if v is None or v == "":
            return dfl
        return str(v)

    return _VAR_DEFAULT.sub(repl, s)


def _expand_mapping(m: Mapping[str, Any], extra: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, str):
            out[str(k)] = expand_str(v, extra)
        elif isinstance(v, dict):
            out[str(k)] = _expand_mapping(v, extra)
        elif isinstance(v, list):
            out[str(k)] = [
                expand_str(x, extra) if isinstance(x, str) else x
                for x in v
            ]
        else:
            out[str(k)] = v
    return out


def registry_path() -> Path:
    p = os.getenv("PHASE_B_EMBEDDING_BASES_YAML", "").strip()
    if p:
        return Path(p)
    return Path(__file__).resolve().parent / "embedding_bases.yaml"


def load_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or registry_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("embedding_bases root must be a mapping")
    data["_path"] = str(p)
    return data


def list_bases(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    bases = data.get("bases")
    if not isinstance(bases, list):
        raise ValueError("bases must be a list")
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for b in bases:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("base_id", "")).strip()
        if not bid:
            raise ValueError("base missing base_id")
        if bid in seen:
            raise ValueError(f"duplicate base_id: {bid}")
        seen.add(bid)
        out.append(b)
    out.sort(key=lambda x: int(x.get("order", 999)))
    return out


def get_base(data: Mapping[str, Any], base_id: str) -> Dict[str, Any]:
    for b in list_bases(data):
        if str(b.get("base_id")) == base_id:
            merged = dict(data.get("defaults") or {})
            merged.update(b)
            return merged
    raise KeyError(base_id)

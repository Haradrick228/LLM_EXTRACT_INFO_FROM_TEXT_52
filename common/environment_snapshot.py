"""Снапшоты окружения для воспроизводимости eval-прогонов.

Каждый прогон записывает в свой каталог ``eval/runs/<ts>_f<N>_<slug>/``:
- ``pip_freeze.txt`` — список установленных пакетов;
- ``env_snapshot.json`` — детерминированный срез ``os.environ`` без секретов;
- ``manifest.json`` — содержит ``git_sha``, ``gold_sha``, версии ключевых пакетов.

Этот модуль выдаёт строительные блоки для всех этих файлов.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Маска для значений: всё, что похоже на ключ/токен/пароль, заменяем на REDACTED.
_SECRET_KEY_PATTERNS = re.compile(
    r"(TOKEN|SECRET|PASSWORD|API_KEY|AUTH|KEY|CREDENTIALS|JWT)",
    re.IGNORECASE,
)

# Какие префиксы env-переменных интересны для RAG/eval контекста.
_INTERESTING_ENV_PREFIXES: tuple[str, ...] = (
    "CHROMA_",
    "RAG_",
    "RERANK_",
    "ST_",
    "RETRIEVAL_",
    "RETRIEVER_",
    "LLM_",
    "OLLAMA_",
    "PHASE_B_",
    "EVAL_",
    "PYTHON",
    "HF_",
    "TRANSFORMERS_",
    "CUDA_",
    "TORCH_",
)


def file_sha256(path: Path, *, chunk_size: int = 65536) -> Optional[str]:
    """SHA-256 файла. Возвращает ``None`` если файла нет."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                buf = f.read(chunk_size)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def git_head_sha(repo_root: Path) -> Optional[str]:
    """Текущий commit SHA. ``None`` если не git или git недоступен."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("ascii", errors="replace").strip()
    except Exception:
        return None


def git_is_dirty(repo_root: Path) -> Optional[bool]:
    """True если есть незакоммиченные правки. ``None`` при ошибке."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return bool(out.strip())
    except Exception:
        return None


def package_versions(packages: Iterable[str]) -> Dict[str, Optional[str]]:
    """Версии конкретных пакетов из текущего окружения."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return {p: None for p in packages}

    out: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def pip_freeze_text() -> str:
    """Вывод ``pip freeze`` (без stderr); пустая строка при ошибке."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def env_snapshot(
    env: Optional[Dict[str, str]] = None,
    *,
    extra_prefixes: tuple[str, ...] = (),
) -> Dict[str, str]:
    """Детерминированный срез env без секретов.

    Берутся переменные с интересными префиксами; значения, имена которых
    похожи на секрет, маскируются как ``"REDACTED"``.
    """
    src = dict(env if env is not None else os.environ)
    prefixes = _INTERESTING_ENV_PREFIXES + tuple(extra_prefixes)
    keys = sorted(k for k in src if k.startswith(prefixes))
    snap: Dict[str, str] = {}
    for k in keys:
        v = src.get(k, "")
        if _SECRET_KEY_PATTERNS.search(k):
            snap[k] = "REDACTED" if v else ""
        else:
            snap[k] = v
    return snap


def build_environment_manifest(
    repo_root: Path,
    *,
    gold_path: Optional[Path] = None,
    track_packages: Iterable[str] = (
        "torch",
        "sentence-transformers",
        "transformers",
        "FlagEmbedding",
        "chromadb",
        "rank-bm25",
        "pymorphy3",
    ),
) -> Dict[str, Any]:
    """Сводный словарь для ``manifest.json`` прогона.

    Содержит: ``python``, ``git``, ``gold``, ``packages``.
    """
    info: Dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "git": {
            "sha": git_head_sha(repo_root),
            "dirty": git_is_dirty(repo_root),
        },
        "packages": package_versions(track_packages),
    }
    if gold_path is not None:
        info["gold"] = {
            "path": str(gold_path),
            "sha256": file_sha256(gold_path),
            "size_bytes": gold_path.stat().st_size if gold_path.exists() else None,
        }
    return info


def write_manifest_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "file_sha256",
    "git_head_sha",
    "git_is_dirty",
    "package_versions",
    "pip_freeze_text",
    "env_snapshot",
    "build_environment_manifest",
    "write_manifest_json",
]

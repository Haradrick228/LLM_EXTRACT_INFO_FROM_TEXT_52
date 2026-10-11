"""Загрузка строк из ``eval/datasets/chunks_sample.jsonl`` (общая фикстура для всех баз фазы B).

Исторический путь ``eval/fixtures/chunks_sample.jsonl`` поддерживается
как fallback (для совместимости со старыми YAML-конфигами).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from eval.text_pipeline import clean_text

_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "chunks_sample.jsonl"
_LEGACY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "chunks_sample.jsonl"


def fixture_path(repo_root: Path | None = None) -> Path:
    """Канонический путь фикстуры: ``eval/datasets/chunks_sample.jsonl``.

    Fallback на ``eval/fixtures/chunks_sample.jsonl`` оставлен для legacy
    YAML-конфигов фазы B (``defaults.fixture_rel``).
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    primary = root / "eval" / "datasets" / "chunks_sample.jsonl"
    if primary.exists():
        return primary
    legacy = root / "eval" / "fixtures" / "chunks_sample.jsonl"
    return legacy if legacy.exists() else primary


def iter_fixture_rows(path: Path | None = None) -> Iterator[Dict[str, Any]]:
    p = path or (_DEFAULT_FIXTURE if _DEFAULT_FIXTURE.exists() else _LEGACY_FIXTURE)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def load_fixture_batches(
    path: Path | None = None,
    *,
    apply_clean_text: bool = True,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Если ``apply_clean_text`` (по умолчанию да): NFKC, удаление управляющих символов, схлопывание пробелов — как ``eval.text_pipeline.clean_text``."""
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    for row in iter_fixture_rows(path):
        ids.append(str(row["id"]))
        raw = str(row["document"])
        docs.append(clean_text(raw) if apply_clean_text else raw)
        metas.append(row.get("metadata") or {})
    return ids, docs, metas

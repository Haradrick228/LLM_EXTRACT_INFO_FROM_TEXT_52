"""Единые пути проекта.

Возвращает абсолютные пути к корню репо, каталогу артефактов прогонов,
датасетам, индексам, конфигам. Использовать ВЕЗДЕ вместо ``Path(__file__).parents[N]``
костылей: это гарантирует, что после перемещения файлов пути не сломаются.
"""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Корень репозитория (содержит ``core/``, ``eval/``, ``DATA/``)."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return project_root() / "DATA"


def runs_dir() -> Path:
    """Каталог immutable артефактов eval-прогонов: ``eval/runs/<ts>_f<N>_<slug>/``."""
    return project_root() / "eval" / "runs"


def indexes_dir() -> Path:
    """Sidecar-индексы (BM25, BGE-M3 lexical/colbert и др.): ``DATA/indexes/``."""
    return data_dir() / "indexes"


def cache_dir() -> Path:
    """Локальные кэши (HyDE-документы, токенизация и т.п.): ``DATA/cache/``."""
    return data_dir() / "cache"


def datasets_dir() -> Path:
    """Каталог с gold/eval-датасетами: ``eval/datasets/``."""
    return project_root() / "eval" / "datasets"


def config_dir() -> Path:
    """YAML-конфиги (embedding_bases, rag_profiles, experiments_registry): ``eval/config/``."""
    return project_root() / "eval" / "config"


def archive_dir() -> Path:
    return project_root() / "archive"


__all__ = [
    "project_root",
    "data_dir",
    "runs_dir",
    "indexes_dir",
    "cache_dir",
    "datasets_dir",
    "config_dir",
    "archive_dir",
]

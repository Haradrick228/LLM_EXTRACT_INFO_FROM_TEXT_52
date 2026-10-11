"""Глобальный pytest conftest.

- Регистрирует маркеры ``unit/integration/regression/gpu`` (включая legacy
  ``slow`` для совместимости).
- Фиксирует ``EVAL_SEED`` и ``RETRIEVAL_EVAL_SHUFFLE=0`` на всю сессию,
  чтобы любой тест, который случайно обращается к этим переменным, видел
  детерминированные значения.
- Подкладывает базовые фикстуры путей и временного ``runs_dir`` через
  :mod:`common.project_paths`.

Для интеграционных тестов с живой Chroma: переменная окружения
``RAG_EVAL_CHROMA_HOST`` (по умолчанию ``127.0.0.1``) и
``RAG_EVAL_CHROMA_PORT`` (по умолчанию ``18000``). Если порт не открыт,
интеграционные тесты помечают себя как ``skip``.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from common import project_paths
from common.reproducibility_seeds import set_global_seed


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "unit: быстрые unit-тесты, без сети/Chroma/GPU.")
    config.addinivalue_line("markers", "integration: требуют живой Chroma/Ollama.")
    config.addinivalue_line("markers", "regression: replay-режим стабильных метрик.")
    config.addinivalue_line("markers", "gpu: требуют CUDA / pynvml.")
    config.addinivalue_line("markers", "slow: legacy alias for regression/integration tests.")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Авто-маркировка тестов по каталогу: ``unit/`` → ``unit`` маркер и т.д."""
    for item in items:
        path = str(item.path) if hasattr(item, "path") else item.fspath  # type: ignore[attr-defined]
        p = str(path).replace("\\", "/")
        if "/tests/unit/" in p:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in p:
            item.add_marker(pytest.mark.integration)
        elif "/tests/regression/" in p:
            item.add_marker(pytest.mark.regression)


@pytest.fixture(autouse=True, scope="session")
def _fix_global_seed() -> None:
    set_global_seed(0)
    os.environ.setdefault("RETRIEVAL_EVAL_SHUFFLE", "0")
    os.environ.setdefault("EVAL_SEED", "0")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return project_paths.project_root()


@pytest.fixture(scope="session")
def datasets_dir() -> Path:
    return project_paths.datasets_dir()


@pytest.fixture(scope="session")
def gold_v2_path(datasets_dir: Path) -> Path:
    return datasets_dir / "eval_set_gold_v2.jsonl"


@pytest.fixture(scope="session")
def chunks_sample_path(datasets_dir: Path) -> Path:
    return datasets_dir / "chunks_sample.jsonl"


@pytest.fixture()
def tmp_runs_dir(tmp_path: Path) -> Path:
    """Изолированный аналог ``eval/runs/`` под одну функцию‑тест."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs


def _tcp_port_open(host: str, port: int, *, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def chroma_available() -> bool:
    host = os.getenv("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("RAG_EVAL_CHROMA_PORT", "18000"))
    except ValueError:
        port = 18000
    return _tcp_port_open(host, port)


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    host = os.getenv("OLLAMA_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("OLLAMA_PORT", "11434"))
    except ValueError:
        port = 11434
    return _tcp_port_open(host, port)

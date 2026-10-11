"""Единый логгер для всего проекта.

Два формата: ``text`` (для человека) и ``json`` (для агрегации артефактов).
Выбирается через env ``RAG_LOG_FORMAT={text,json}``, по умолчанию ``text``.

Любой модуль получает logger через ``get_logger(__name__)``. Дополнительные
поля (``phase``, ``scenario``, ``run_id``, ``event``, ``took_ms``) передаются
как kwargs через ``logger.info("event_name", extra={"phase":...})`` или через
helper ``log_event(logger, "event", phase=..., took_ms=...)``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

_RESERVED_LOGRECORD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class _JsonFormatter(logging.Formatter):
    """Структурированный JSON-формат: одна строка = один JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in _RESERVED_LOGRECORD_FIELDS:
                continue
            if k.startswith("_"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Человекочитаемый формат с расширенными полями в скобках."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} {record.levelname:5s} {record.name}: {record.getMessage()}"
        extras = []
        for k, v in record.__dict__.items():
            if k in _RESERVED_LOGRECORD_FIELDS or k.startswith("_"):
                continue
            extras.append(f"{k}={v}")
        if extras:
            base += "  [" + " ".join(extras) + "]"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def _resolve_level() -> int:
    raw = (os.getenv("RAG_LOG_LEVEL") or "INFO").upper()
    return getattr(logging, raw, logging.INFO)


def _resolve_format() -> str:
    raw = (os.getenv("RAG_LOG_FORMAT") or "text").strip().lower()
    return "json" if raw == "json" else "text"


_ROOT_CONFIGURED = False


def configure_root_logger(*, stream=None, force: bool = False) -> None:
    """Однократно настраивает root‑логгер. Идемпотентно."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED and not force:
        return
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_JsonFormatter() if _resolve_format() == "json" else _TextFormatter())
    root.addHandler(handler)
    root.setLevel(_resolve_level())
    _ROOT_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Получить логгер. Гарантирует, что root настроен."""
    configure_root_logger()
    return logging.getLogger(name if name else "rag")


def log_event(logger: logging.Logger, event: str, /, *, level: int = logging.INFO, **fields: Any) -> None:
    """Залогировать структурированное событие.

    ``event`` — короткое имя действия (``dense.query``, ``rerank.start``).
    ``fields`` — произвольные ключи, пишутся в `extra` и попадают в JSON/text.
    """
    logger.log(level, event, extra={"event": event, **fields})


@contextmanager
def timed(logger: logging.Logger, event: str, /, **fields: Any):
    """Контекст‑менеджер: логирует ``event_started`` и ``event`` с ``took_ms``."""
    log_event(logger, f"{event}_started", **fields)
    t0 = time.monotonic()
    try:
        yield
    finally:
        took_ms = int((time.monotonic() - t0) * 1000)
        log_event(logger, event, took_ms=took_ms, **fields)


__all__ = [
    "configure_root_logger",
    "get_logger",
    "log_event",
    "timed",
]

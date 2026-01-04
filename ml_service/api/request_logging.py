import time
from typing import Any, Dict, List, Optional

try:
    from core.db.session import session_scope
    from core.db.repositories.request_log import RequestLogRepository
except Exception:
    from contextlib import contextmanager

    @contextmanager
    def session_scope():
        yield None

    class RequestLogRepository: 
        def __init__(self, session=None):
            pass

        def add(self, **kwargs):
            return None

        def list(self, limit: int = 200):
            return []

        def clear_all(self):
            return 0

        def stats(self):
            return {}

__all__ = ["log_request", "get_history", "clear_history", "get_stats"]


def log_request(
    *,
    start: float,
    input_type: str,
    model: Optional[str],
    status: str,
    error: Optional[str],
    text_len: Optional[int],
    token_count: Optional[int],
    image_width: Optional[int],
    image_height: Optional[int],
    response_preview: Optional[str],
) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    with session_scope() as session:
        repo = RequestLogRepository(session)
        repo.add(
            input_type=input_type,
            model=model,
            status=status,
            error=error,
            duration_ms=duration_ms,
            text_len=text_len,
            token_count=token_count,
            image_width=image_width,
            image_height=image_height,
            response_preview=response_preview,
        )


def get_history(limit: int = 200) -> List[Dict[str, Any]]:
    with session_scope() as session:
        repo = RequestLogRepository(session)
        rows = repo.list(limit=limit)
        return [
            {
                "id": r.id,
                "created_at": r.created_at,
                "input_type": r.input_type,
                "model": r.model,
                "status": r.status,
                "error": r.error,
                "duration_ms": r.duration_ms,
                "text_len": r.text_len,
                "token_count": r.token_count,
                "image_width": r.image_width,
                "image_height": r.image_height,
                "response_preview": r.response_preview,
            }
            for r in rows
        ]


def clear_history() -> int:
    with session_scope() as session:
        repo = RequestLogRepository(session)
        return repo.clear_all()


def get_stats() -> Dict[str, Any]:
    with session_scope() as session:
        repo = RequestLogRepository(session)
        return repo.stats()

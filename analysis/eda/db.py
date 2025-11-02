from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import text

from core.db_service import DBService
from core.db.session import get_engine

from .models import CleanMetadataUpdate

LOGGER = logging.getLogger("analysis.eda.db")


class DatabaseClient:
    def __init__(self, service: Optional[DBService] = None) -> None:
        self._service = service or DBService()
        self.available = True
        self._metadata_cache: Dict[str, Optional[dict]] = {}
        self._scanned_cache: Dict[str, Optional[dict]] = {}

    def ensure_clean_columns(self) -> None:
        if not self.available:
            return
        statements = (
            "ALTER TABLE public.file_metadata ADD COLUMN IF NOT EXISTS clean_text_path text",
            "ALTER TABLE public.file_metadata ADD COLUMN IF NOT EXISTS clean_text_hash varchar(128)",
            "ALTER TABLE public.file_metadata ADD COLUMN IF NOT EXISTS cleaned_at timestamptz",
            "ALTER TABLE public.scanned_pages ADD COLUMN IF NOT EXISTS clean_text_path text",
            "ALTER TABLE public.scanned_pages ADD COLUMN IF NOT EXISTS clean_text_hash varchar(128)",
            "ALTER TABLE public.scanned_pages ADD COLUMN IF NOT EXISTS cleaned_at timestamptz",
        )
        try:
            engine = get_engine()
        except Exception as exc:
            LOGGER.warning("Не удалось получить соединение с БД: %s", exc)
            self.available = False
            return

        try:
            with engine.begin() as connection:
                for stmt in statements:
                    connection.execute(text(stmt))
        except Exception as exc:
            LOGGER.warning("Не удалось актуализировать схему БД: %s", exc)
            self.available = False

    def fetch_classification_summary(self) -> Dict[str, List[Tuple[str, int]]]:
        if not self.available:
            return {}
        try:
            engine = get_engine()
        except Exception as exc:
            LOGGER.warning("Не удалось подключиться к БД для получения статистики классификации: %s", exc)
            self.available = False
            return {}

        queries = {
            "content_type": """
                SELECT COALESCE(content_type, 'unknown') AS label, COUNT(*)::int AS total
                FROM public.file_classification
                GROUP BY 1
                ORDER BY total DESC
            """,
            "education_level": """
                SELECT COALESCE(education_level, 'unknown') AS label, COUNT(*)::int AS total
                FROM public.file_classification
                GROUP BY 1
                ORDER BY total DESC
            """,
            "audience": """
                SELECT COALESCE(audience, 'unknown') AS label, COUNT(*)::int AS total
                FROM public.file_classification
                GROUP BY 1
                ORDER BY total DESC
            """,
            "purpose": """
                SELECT COALESCE(purpose, 'unknown') AS label, COUNT(*)::int AS total
                FROM public.file_classification
                GROUP BY 1
                ORDER BY total DESC
            """,
        }

        summary: Dict[str, List[Tuple[str, int]]] = {}
        try:
            with engine.connect() as conn:
                for label, query in queries.items():
                    result = conn.execute(text(query))
                    summary[label] = [(row.label, int(row.total)) for row in result]
        except Exception as exc:
            LOGGER.warning("Не удалось получить статистику классификации: %s", exc)
            self.available = False
            return {}

        return summary

    def update_clean_metadata(self, file_name: str, output_path: Path, content_hash: str) -> CleanMetadataUpdate:
        if not self.available:
            return CleanMetadataUpdate(linked=False, available=False, message="db_unavailable")

        cleaned_ts = datetime.now(timezone.utc)
        cleaned_path_str = output_path.as_posix()
        found_target = False

        metadata_entry = self._get_metadata(file_name)
        if self.available and metadata_entry and metadata_entry.get("id"):
            try:
                self._service.update_file_clean_metadata(
                    file_metadata_id=metadata_entry["id"],
                    clean_text_path=cleaned_path_str,
                    clean_text_hash=content_hash,
                    cleaned_at=cleaned_ts,
                )
                found_target = True
            except Exception as exc:
                LOGGER.warning("Не удалось связать файл %s с file_metadata: %s", file_name, exc)

        if not found_target and self.available:
            scanned_entry = self._get_scanned_page(file_name)
            if scanned_entry and scanned_entry.get("id"):
                try:
                    self._service.update_scanned_page_clean_metadata(
                        scanned_page_id=scanned_entry["id"],
                        clean_text_path=cleaned_path_str,
                        clean_text_hash=content_hash,
                        cleaned_at=cleaned_ts,
                    )
                    found_target = True
                except Exception as exc:
                    LOGGER.warning("Не удалось связать файл %s с scanned_pages: %s", file_name, exc)

        if not self.available:
            return CleanMetadataUpdate(linked=False, available=False, message="db_unavailable")
        if found_target:
            return CleanMetadataUpdate(linked=True, available=True)
        return CleanMetadataUpdate(linked=False, available=True, message="not_found")

    def _get_metadata(self, file_name: str) -> Optional[dict]:
        if file_name not in self._metadata_cache:
            try:
                self._metadata_cache[file_name] = self._service.find_file_metadata_by_name(file_name)
            except Exception as exc:
                LOGGER.warning("Не удалось получить метаданные file_metadata для %s: %s", file_name, exc)
                self._metadata_cache[file_name] = None
                self.available = False
        return self._metadata_cache.get(file_name)

    def _get_scanned_page(self, file_name: str) -> Optional[dict]:
        if file_name not in self._scanned_cache:
            try:
                self._scanned_cache[file_name] = self._service.find_scanned_page_by_pdf(file_name)
            except Exception as exc:
                LOGGER.warning("Не удалось получить запись scanned_pages для %s: %s", file_name, exc)
                self._scanned_cache[file_name] = None
                self.available = False
        return self._scanned_cache.get(file_name)

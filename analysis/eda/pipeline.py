from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

from tqdm import tqdm

from common.util import extract_text_from_file

from .config import (
    DEFAULT_TOP_TERMS,
    LONG_DOC_TOKENS,
    SHORT_DOC_TOKENS,
    SUPPORTED_EXTENSIONS,
)
from .db import DatabaseClient
from .filesystem import ensure_write, iter_documents
from .models import DocumentRecord, PipelineSettings, ProcessingError, ReportPaths
from .reporting import DashboardRenderer
from .stats import build_tfidf_report, compute_term_coverage, describe
from .text import TextProcessor

LOGGER = logging.getLogger("analysis.eda.pipeline")


class EDAPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        db_client: Optional[DatabaseClient] = None,
        text_processor: Optional[TextProcessor] = None,
    ) -> None:
        self.settings = settings
        self.db = db_client or DatabaseClient()
        self.text = text_processor or TextProcessor()
        self.renderer = DashboardRenderer()

    def run(self) -> ReportPaths:
        self.db.ensure_clean_columns()

        documents = list(iter_documents(self.settings.input_dirs, SUPPORTED_EXTENSIONS))
        total_docs = len(documents)
        if self.settings.max_files is not None:
            documents = documents[: self.settings.max_files]

        LOGGER.info("Найдено %s документов (%s с учётом ограничения).", total_docs, len(documents))

        aggregation = self._process_documents(documents)
        report = self._build_report(total_docs, aggregation)

        self.settings.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.settings.report_dir / "eda_report.json"
        dashboard_path = self.settings.report_dir / "eda_dashboard.html"

        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self.renderer.render(report, dashboard_path)

        LOGGER.info("JSON-отчёт записан в %s", report_path)
        LOGGER.info("HTML-дашборд записан в %s", dashboard_path)

        return ReportPaths(json_report=report_path, html_dashboard=dashboard_path)

    def _process_documents(
        self,
        documents: Sequence[Tuple[Path, Path]],
    ) -> Dict[str, object]:
        unique_hashes: Dict[str, DocumentRecord] = {}
        duplicates: Dict[str, List[str]] = defaultdict(list)
        vocabulary: Counter = Counter()
        language_counts: Counter = Counter()
        doc_char_lengths: List[int] = []
        doc_token_lengths: List[int] = []
        chunk_lengths: List[int] = []
        chunk_counts: List[int] = []
        unique_texts: Dict[str, str] = {}
        errors: List[ProcessingError] = []
        linked_hashes: set[str] = set()

        cleaned_doc_count = 0
        short_doc_count = 0
        long_doc_count = 0

        for base_dir, path in tqdm(documents, desc="Обработка документов", unit="док."):
            try:
                raw_text = extract_text_from_file(str(path))
            except Exception as exc:
                errors.append(ProcessingError(path=str(path), error=f"extract_failed: {exc}"))
                LOGGER.warning("Не удалось извлечь текст из %s: %s", path, exc)
                continue

            cleaned = self.text.clean(raw_text)
            if not cleaned:
                errors.append(ProcessingError(path=str(path), error="empty_after_clean"))
                LOGGER.debug("Документ %s пуст после очистки.", path)
                continue

            content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            rel_path = path.relative_to(base_dir)
            output_path = (self.settings.output_dir / rel_path).with_suffix(".txt") if self.settings.output_dir else None

            existing = unique_hashes.get(content_hash)
            if existing:
                duplicates[existing.source_path.as_posix()].append(str(path))
                LOGGER.debug("Найден дубликат: %s -> %s", path, existing.source_path)
                continue

            tokens = self.text.tokenize(cleaned)
            token_count = len(tokens)
            language = self.text.detect_language(cleaned)
            chunk_list = self.text.split_into_chunks(cleaned)

            vocabulary.update(tokens)
            language_counts[language] += 1
            doc_char_lengths.append(len(cleaned))
            doc_token_lengths.append(token_count)
            chunk_counts.append(len(chunk_list))
            chunk_lengths.extend(len(chunk) for chunk in chunk_list)

            if token_count <= SHORT_DOC_TOKENS:
                short_doc_count += 1
            if token_count >= LONG_DOC_TOKENS:
                long_doc_count += 1

            record = DocumentRecord(
                source_path=path,
                output_path=output_path,
                content_hash=content_hash,
                characters=len(cleaned),
                tokens=token_count,
                chunks=len(chunk_list),
                language=language,
                has_non_ascii=self.text.has_non_ascii(cleaned),
            )
            unique_hashes[content_hash] = record
            unique_texts[content_hash] = cleaned

            ensure_write(output_path, cleaned, self.settings.overwrite)

            if output_path:
                cleaned_doc_count += 1
                if self.db.available:
                    update = self.db.update_clean_metadata(path.name, output_path, content_hash)
                    if update.linked:
                        linked_hashes.add(content_hash)
                    elif not update.available:
                        LOGGER.debug("Пропускаем связывание %s: база недоступна.", path.name)
                else:
                    LOGGER.debug("Пропускаем связывание %s: база недоступна.", path.name)

        linked_clean_docs = len(linked_hashes)
        metadata_missing_count = max(cleaned_doc_count - linked_clean_docs, 0)

        classification_summary = self.db.fetch_classification_summary() if self.db.available else {}

        return {
            "documents_total": len(documents),
            "unique_hashes": unique_hashes,
            "duplicates": duplicates,
            "vocabulary": vocabulary,
            "language_counts": language_counts,
            "doc_char_lengths": doc_char_lengths,
            "doc_token_lengths": doc_token_lengths,
            "chunk_lengths": chunk_lengths,
            "chunk_counts": chunk_counts,
            "unique_texts": unique_texts,
            "errors": errors,
            "cleaned_doc_count": cleaned_doc_count,
            "linked_clean_docs": linked_clean_docs,
            "metadata_missing_count": metadata_missing_count,
            "short_doc_count": short_doc_count,
            "long_doc_count": long_doc_count,
            "linked_hashes": linked_hashes,
            "classification_summary": classification_summary,
        }

    def _build_report(self, total_docs: int, aggregation: Dict[str, object]) -> Dict[str, object]:
        unique_hashes = cast(Dict[str, DocumentRecord], aggregation["unique_hashes"])
        duplicates = cast(Dict[str, List[str]], aggregation["duplicates"])
        vocabulary = cast(Counter, aggregation["vocabulary"])
        language_counts = cast(Counter, aggregation["language_counts"])
        doc_char_lengths = cast(List[int], aggregation["doc_char_lengths"])
        doc_token_lengths = cast(List[int], aggregation["doc_token_lengths"])
        chunk_lengths = cast(List[int], aggregation["chunk_lengths"])
        chunk_counts = cast(List[int], aggregation["chunk_counts"])
        unique_texts = cast(Dict[str, str], aggregation["unique_texts"])
        errors = cast(List[ProcessingError], aggregation["errors"])
        cleaned_doc_count = cast(int, aggregation["cleaned_doc_count"])
        linked_clean_docs = cast(int, aggregation["linked_clean_docs"])
        metadata_missing_count = cast(int, aggregation["metadata_missing_count"])
        short_doc_count = cast(int, aggregation["short_doc_count"])
        long_doc_count = cast(int, aggregation["long_doc_count"])
        linked_hashes = cast(set[str], aggregation["linked_hashes"])
        classification_summary = cast(Dict[str, List[Tuple[str, int]]], aggregation["classification_summary"])

        duplicate_total = sum(len(paths) for paths in duplicates.values())
        errors_count = len(errors)
        unique_docs = len(unique_hashes)
        processed_docs = cast(int, aggregation["documents_total"])
        non_ascii_count = sum(1 for record in unique_hashes.values() if record.has_non_ascii)
        total_tokens = sum(doc_token_lengths)
        total_chars = sum(doc_char_lengths)
        total_chunks = sum(chunk_counts)
        unique_token_count = len(vocabulary)

        duplicate_ratio = duplicate_total / processed_docs if processed_docs else 0.0
        error_rate = errors_count / processed_docs if processed_docs else 0.0
        non_ascii_ratio = non_ascii_count / unique_docs if unique_docs else 0.0
        avg_tokens_per_doc = total_tokens / unique_docs if unique_docs else 0.0
        avg_chars_per_doc = total_chars / unique_docs if unique_docs else 0.0
        avg_chunks_per_doc = total_chunks / unique_docs if unique_docs else 0.0
        avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks else 0.0
        lexical_diversity = unique_token_count / total_tokens if total_tokens else 0.0
        coverage = compute_term_coverage(self.settings.terms, vocabulary)
        coverage_value = coverage.get("coverage", 0.0)
        clean_path_coverage = linked_clean_docs / unique_docs if unique_docs else 0.0
        linked_clean_ratio = linked_clean_docs / cleaned_doc_count if cleaned_doc_count else 0.0
        metadata_gap_ratio = metadata_missing_count / cleaned_doc_count if cleaned_doc_count else 0.0
        short_doc_ratio = short_doc_count / unique_docs if unique_docs else 0.0
        long_doc_ratio = long_doc_count / unique_docs if unique_docs else 0.0
        dominant_language_label = "unknown"
        dominant_language_share = 0.0
        if language_counts and unique_docs:
            dominant_language_label, dominant_count = max(language_counts.items(), key=lambda item: item[1])
            dominant_language_share = dominant_count / unique_docs
        largest_duplicate_group_size = max((len(paths) for paths in duplicates.values()), default=0)
        largest_duplicate_group_docs = largest_duplicate_group_size + 1 if largest_duplicate_group_size else 0

        top_terms = vocabulary.most_common(DEFAULT_TOP_TERMS)
        tfidf_top = build_tfidf_report(unique_texts, DEFAULT_TOP_TERMS)

        duplicate_groups = sorted(
            ((source, paths) for source, paths in duplicates.items()),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        top_duplicate_groups = [
            {
                "canonical": source,
                "duplicates": len(paths),
                "examples": paths[:5],
            }
            for source, paths in duplicate_groups[:10]
        ]

        error_samples = [{"path": err.path, "error": err.error} for err in errors[:10]]

        quality_metrics = {
            "duplicate_ratio": {
                "label": "Доля дубликатов",
                "value": duplicate_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Отношение числа дубликатов к числу обработанных документов.",
            },
            "error_rate": {
                "label": "Ошибки обработки",
                "value": error_rate,
                "format": "percent",
                "digits": 2,
                "description": "Часть документов, завершившихся ошибкой извлечения или очистки.",
            },
            "clean_text_coverage": {
                "label": "Связанные тексты",
                "value": clean_path_coverage,
                "format": "percent",
                "digits": 2,
                "description": "Доля уникальных документов, для которых получен и сохранён очищенный текст.",
            },
            "clean_text_linked_ratio": {
                "label": "Связаны с БД",
                "value": linked_clean_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Доля сохранённых текстов, у которых обновлены привязки в file_metadata или scanned_pages.",
            },
            "metadata_gap_ratio": {
                "label": "Без связи в БД",
                "value": metadata_gap_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Доля сохранённых текстов, для которых не найдено соответствующей записи в базе.",
            },
            "short_doc_ratio": {
                "label": f"Короткие документы (≤ {SHORT_DOC_TOKENS} токенов)",
                "value": short_doc_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Часть корпуса, потенциально требующая агрегации или исключения из индекса.",
            },
            "long_doc_ratio": {
                "label": f"Длинные документы (≥ {LONG_DOC_TOKENS} токенов)",
                "value": long_doc_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Часть корпуса, требующая сегментации или особого режима обработки.",
            },
            "avg_tokens_per_document": {
                "label": "Средняя длина (токены)",
                "value": avg_tokens_per_doc,
                "format": "number",
                "digits": 0,
                "description": "Среднее число токенов на документ.",
            },
            "avg_chars_per_document": {
                "label": "Средняя длина (символы)",
                "value": avg_chars_per_doc,
                "format": "number",
                "digits": 0,
                "description": "Среднее число символов на документ после очистки.",
            },
            "avg_chunks_per_document": {
                "label": "Среднее число чанков",
                "value": avg_chunks_per_doc,
                "format": "number",
                "digits": 2,
                "description": "Сколько фрагментов в среднем формируется из одного документа.",
            },
            "avg_tokens_per_chunk": {
                "label": "Средняя длина чанка",
                "value": avg_tokens_per_chunk,
                "format": "number",
                "digits": 1,
                "description": "Среднее число токенов в чанке.",
            },
            "lexical_diversity": {
                "label": "Лексическое разнообразие",
                "value": lexical_diversity,
                "format": "percent",
                "digits": 2,
                "description": "Отношение количества уникальных токенов к общему числу токенов.",
            },
            "dominant_language_share": {
                "label": f"Доминирующий язык ({dominant_language_label})",
                "value": dominant_language_share,
                "format": "percent",
                "digits": 1,
                "description": "Доля документов на самом распространённом языке.",
            },
            "unique_token_count": {
                "label": "Количество уникальных токенов",
                "value": unique_token_count,
                "format": "integer",
                "digits": 0,
                "description": "Общее число уникальных токенов в корпусе.",
            },
            "largest_duplicate_group": {
                "label": "Крупнейшая группа дубликатов",
                "value": largest_duplicate_group_docs,
                "format": "integer",
                "digits": 0,
                "description": "Размер самой большой группы взаимных дубликатов.",
            },
            "non_ascii_ratio": {
                "label": "Содержат не-ASCII символы",
                "value": non_ascii_ratio,
                "format": "percent",
                "digits": 2,
                "description": "Доля документов с символами вне ASCII (может указывать на мусор или HTML).",
            },
        }

        legend = {
            "documents.total_candidates": "Документы, найденные в указанных директориях.",
            "documents.processed": "Файлы, рассмотренные после применения лимита max-files (если задан).",
            "documents.unique": "Уникальные по содержимому документы.",
            "documents.duplicates": "Количество дубликатов, отфильтрованных по хешу.",
            "documents.cleaned_with_output": "Документы, для которых сохранён очищенный текст.",
            "documents.linked_clean_documents": "Документы, привязанные к file_metadata или scanned_pages.",
            "documents.unlinked_clean_documents": "Сохранённые тексты без найденной записи в базе.",
            "documents.short_documents": "Документы с длиной ≤ заданного минимума.",
            "documents.long_documents": "Документы с длиной ≥ заданного максимума.",
            "documents.language_distribution": "Распределение документов по языкам.",
            "documents.char_stats": "Статистика по числу символов.",
            "documents.token_stats": "Статистика по числу токенов.",
            "documents.chunk_count_stats": "Статистика по числу чанков.",
            "documents.chunk_size_stats": "Статистика по размеру чанков.",
            "top_terms": "Самые частые термины в корпусе.",
            "tfidf_top_features": "Термины с максимальным средним TF-IDF.",
            "term_coverage": "Покрытие пользовательского словаря в корпусе.",
            "quality_metrics.clean_text_coverage": "Доля уникальных документов с сохранённым текстом.",
            "quality_metrics.clean_text_linked_ratio": "Доля сохранённых текстов с привязкой к БД.",
            "quality_metrics.metadata_gap_ratio": "Сохранённые тексты без записи в БД.",
            "quality_metrics.short_doc_ratio": "Доля коротких документов.",
            "quality_metrics.long_doc_ratio": "Доля длинных документов.",
            "quality_metrics.dominant_language_share": "Доля доминирующего языка.",
            "quality_metrics.largest_duplicate_group": "Размер крупнейшей группы дубликатов.",
            "classification_summary.content_type": "Распределение документов по типу контента.",
            "classification_summary.education_level": "Распределение по уровню образования.",
            "classification_summary.audience": "Распределение по целевой аудитории.",
            "classification_summary.purpose": "Распределение по назначению документа.",
            "quality_metrics": "Ключевые метрики качества загрузки корпуса.",
        }

        document_details = [
            {
                "source": record.source_path.as_posix(),
                "normalized": record.output_path.as_posix() if record.output_path else None,
                "hash": record.content_hash,
                "clean_text_path": record.output_path.as_posix() if record.output_path else None,
                "clean_text_hash": record.content_hash,
                "characters": record.characters,
                "tokens": record.tokens,
                "chunks": record.chunks,
                "language": record.language,
                "has_non_ascii": record.has_non_ascii,
                "linked_in_db": record.content_hash in linked_hashes,
            }
            for record in unique_hashes.values()
        ]

        generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        report = {
            "generated_at_utc": generated_at,
            "documents": {
                "total_candidates": total_docs,
                "processed": processed_docs,
                "unique": unique_docs,
                "duplicates": duplicate_total,
                "duplicate_ratio": duplicate_ratio,
                "language_distribution": dict(language_counts),
                "documents_with_non_ascii": non_ascii_count,
                "cleaned_with_output": cleaned_doc_count,
                "linked_clean_documents": linked_clean_docs,
                "unlinked_clean_documents": metadata_missing_count,
                "short_documents": short_doc_count,
                "long_documents": long_doc_count,
                "char_stats": describe(doc_char_lengths),
                "token_stats": describe(doc_token_lengths),
                "chunk_count_stats": describe(chunk_counts),
                "chunk_size_stats": describe(chunk_lengths),
            },
            "quality_metrics": quality_metrics,
            "legend": legend,
            "top_terms": top_terms,
            "tfidf_top_features": tfidf_top,
            "term_coverage": coverage,
            "top_duplicate_groups": top_duplicate_groups,
            "duplicates": {key: value for key, value in duplicates.items()},
            "document_details": document_details,
            "errors": [{"path": err.path, "error": err.error} for err in errors],
            "error_samples": error_samples,
            "output_dir": str(self.settings.output_dir) if self.settings.output_dir else None,
            "dashboard_path": (self.settings.report_dir / "eda_dashboard.html").as_posix(),
            "db_available": self.db.available,
            "classification_summary": classification_summary,
        }
        return report

from __future__ import annotations

import logging
import textwrap
from html import escape
from pathlib import Path
from typing import Dict, List

from .config import DASHBOARD_CSS_PATH
from .formatters import format_metric_value, format_number, format_percentage

LOGGER = logging.getLogger("analysis.eda.reporting")


class DashboardRenderer:
    def __init__(self, css_path: Path = DASHBOARD_CSS_PATH) -> None:
        self.css_path = css_path

    def render(self, report: Dict[str, object], dashboard_path: Path) -> None:
        docs = report.get("documents", {})
        metrics: Dict[str, Dict[str, object]] = report.get("quality_metrics", {})
        languages: Dict[str, int] = docs.get("language_distribution", {})
        top_terms = report.get("top_terms", [])
        tfidf_top = report.get("tfidf_top_features", [])
        top_duplicates = report.get("top_duplicate_groups", [])
        coverage = report.get("term_coverage", {})
        error_samples = report.get("error_samples", [])
        generated_at = report.get("generated_at_utc", "")
        db_available = bool(report.get("db_available", True))
        class_summary = report.get("classification_summary", {})

        summary_data = [
            ("Всего кандидатов", docs.get("total_candidates", 0)),
            ("Обработано документов", docs.get("processed", 0)),
            ("Уникальные тексты", docs.get("unique", 0)),
            ("Дубликаты", docs.get("duplicates", 0)),
            ("Сохранены тексты", docs.get("cleaned_with_output", 0)),
            ("Связаны с БД", docs.get("linked_clean_documents", 0)),
            ("Остались без связи", docs.get("unlinked_clean_documents", 0)),
            ("Ошибки обработки", len(report.get("errors", []))),
        ]
        summary_cards = "".join(
            f"""
            <div class="card">
                <div class="value">{format_number(value, 0)}</div>
                <div class="label">{escape(label)}</div>
            </div>
            """
            for label, value in summary_data
        )

        unique_docs = docs.get("unique", 0) or 0
        dominant_lang = "unknown"
        dominant_share = 0.0
        if languages and unique_docs:
            dominant_lang, dominant_count = max(languages.items(), key=lambda item: item[1])
            dominant_share = dominant_count / unique_docs if unique_docs else 0.0
        duplicate_ratio = metrics.get("duplicate_ratio", {}).get("value")
        largest_dup = metrics.get("largest_duplicate_group", {}).get("value")
        clean_coverage = metrics.get("clean_text_coverage", {}).get("value")
        linked_ratio = metrics.get("clean_text_linked_ratio", {}).get("value")
        metadata_gap = metrics.get("metadata_gap_ratio", {}).get("value")
        error_rate = metrics.get("error_rate", {}).get("value")
        term_cov = coverage.get("coverage")
        short_ratio = metrics.get("short_doc_ratio", {}).get("value")
        long_ratio = metrics.get("long_doc_ratio", {}).get("value")

        summary_points: List[str] = []
        summary_points.append(
            f"Обработано {format_number(docs.get('processed', 0), 0)} документов, уникальных текстов — "
            f"{format_number(unique_docs, 0)}."
        )
        if not db_available:
            summary_points.append(
                "База данных недоступна: связывание текстов с file_metadata и scanned_pages не выполнялось."
            )
        if duplicate_ratio is not None:
            text = f"Доля дубликатов — {format_percentage(duplicate_ratio, 2)}"
            if largest_dup:
                text += f"; крупнейшая группа насчитывает {format_number(largest_dup, 0)} файлов."
            summary_points.append(text + ".")
        if clean_coverage is not None and linked_ratio is not None:
            summary_points.append(
                f"Ссылки на очищенные тексты есть у {format_percentage(clean_coverage, 2)} документов; "
                f"{format_percentage(linked_ratio, 2)} из них связаны с БД."
            )
        if metadata_gap is not None and metadata_gap > 0:
            summary_points.append(
                f"Не удалось связать {format_percentage(metadata_gap, 2)} сохранённых текстов с БД."
            )
        if error_rate is not None:
            summary_points.append(f"Ошибками завершилось {format_percentage(error_rate, 2)} обработок.")
        if term_cov is not None:
            summary_points.append(
                f"Целевые термины покрыты на {format_percentage(term_cov, 1)} "
                f"(коротких: {format_percentage(short_ratio or 0.0, 1)}, длинных: {format_percentage(long_ratio or 0.0, 1)})."
            )
        if dominant_share:
            summary_points.append(
                f"Доминирующий язык — {escape(dominant_lang)}, доля {format_percentage(dominant_share, 1)}."
            )
        summary_html = "".join(f"<li>{point}</li>" for point in summary_points)

        metric_rows = "".join(
            f"""
            <tr>
                <td class="metric-name">{escape(metric.get('label', ''))}</td>
                <td class="metric-value">{format_metric_value(metric)}</td>
                <td class="metric-desc">{escape(str(metric.get('description', '')))}</td>
            </tr>
            """
            for metric in metrics.values()
        )

        language_rows = "".join(
            f"""
            <tr>
                <td>{escape(lang)}</td>
                <td class="center">{format_number(total, 0)}</td>
                <td class="center">{format_percentage((total or 0) / unique_docs, 1) if unique_docs else '—'}</td>
                <td>
                    <div class="bar"><span style="width: {min((total or 0) / unique_docs * 100, 100):.1f}%"></span></div>
                </td>
            </tr>
            """
            for lang, total in languages.items()
        )

        top_term_rows = "".join(
            f"<tr><td>{escape(term)}</td><td class=\"center\">{format_number(freq, 0)}</td></tr>"
            for term, freq in top_terms
        )
        tfidf_rows = "".join(
            f"<tr><td>{escape(term)}</td><td class=\"center\">{freq:.4f}</td></tr>"
            for term, freq in tfidf_top
        )

        classification_cards: List[str] = []
        for label, rows in class_summary.items():
            if not rows:
                continue
            table_rows = "".join(
                f"<tr><td>{escape(name)}</td><td class=\"center\">{format_number(total, 0)}</td></tr>"
                for name, total in rows[:15]
            )
            classification_cards.append(
                f"""
                <div class="card">
                    <h3>{escape(label)}</h3>
                    <table>
                        <thead><tr><th>Категория</th><th class="center">Документов</th></tr></thead>
                        <tbody>{table_rows}</tbody>
                    </table>
                </div>
                """
            )

        if classification_cards:
            classification_html = "<div class=\"cards\">" + "".join(classification_cards) + "</div>"
        else:
            if db_available:
                classification_html = "<p>Связанные классификационные данные отсутствуют.</p>"
            else:
                classification_html = "<p>Связанные классификационные данные недоступны без подключения к БД.</p>"
        classification_section = f"""
            <section>
                <h2>Сводка классификаций</h2>
                {classification_html}
            </section>
            """

        coverage_present = coverage.get("present", []) or []
        coverage_missing = coverage.get("missing", []) or []
        coverage_total = len(coverage_present) + len(coverage_missing)
        coverage_block = ""
        if coverage_total:
            coverage_block = f"""
            <p>
                Найдено целевых терминов: {format_number(len(coverage_present), 0)} из {format_number(coverage_total, 0)}
                ({format_percentage(coverage.get('coverage', 0.0), 1)}).
            </p>
            """
            if coverage_missing:
                missing_list = "".join(f"<li>{escape(term)}</li>" for term in coverage_missing)
                coverage_block += f"""
                <details>
                    <summary>Не найдены ({len(coverage_missing)})</summary>
                    <ul class="term-list">{missing_list}</ul>
                </details>
                """

        duplicates_rows = "".join(
            f"""
            <tr>
                <td>{escape(group.get('canonical', ''))}</td>
                <td class="center">{format_number(group.get('duplicates', 0), 0)}</td>
                <td>
                    <ul class="summary-list">
                        {''.join(f"<li>{escape(path)}</li>" for path in group.get('examples', []))}
                    </ul>
                </td>
            </tr>
            """
            for group in top_duplicates
        ) or "<tr><td colspan='3' class='center'>Дубликаты не обнаружены.</td></tr>"

        error_rows = "".join(
            f"<tr><td>{escape(item.get('path', ''))}</td><td>{escape(item.get('error', ''))}</td></tr>"
            for item in error_samples
        ) or "<tr><td colspan='2'>Ошибок обработки не зафиксировано.</td></tr>"

        try:
            styles = self.css_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            LOGGER.warning("CSS файл %s не найден, используем базовые стили.", self.css_path)
            styles = ""
        styles_block = textwrap.indent(styles.strip(), "        ")

        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>EDA-дашборд корпуса документов</title>
    <style>
{styles_block}
    </style>
</head>
<body>
    <header>
        <h1>Exploratory Data Analysis корпуса документов</h1>
        <p>Отчёт сгенерирован: {escape(str(generated_at))}</p>
    </header>
    <main>
        <section>
            <h2>Ключевые цифры</h2>
            <div class="cards">
                {summary_cards}
            </div>
        </section>
        <section>
            <h2>Показатели качества загрузки</h2>
            <p>Метрики характеризуют поток обработки, покрытие базы и актуальность связей с данными.</p>
            <table>
                <thead>
                    <tr><th>Показатель</th><th>Значение</th><th>Комментарий</th></tr>
                </thead>
                <tbody>
                    {metric_rows}
                </tbody>
            </table>
        </section>
        <section>
            <h2>Языки корпуса</h2>
            <p>Распределение по языкам помогает определить объём многоязычных материалов.</p>
            <table>
                <thead>
                    <tr><th>Язык</th><th class="center">Документов</th><th class="center">Доля</th><th>Вклад</th></tr>
                </thead>
                <tbody>
                    {language_rows or '<tr><td colspan="4">Языковое распределение не вычислено.</td></tr>'}
                </tbody>
            </table>
        </section>
        <section>
            <h2>Лексический состав</h2>
            <div class="cards">
                <div class="card">
                    <h3>Частотный словарь</h3>
                    <table>
                        <thead><tr><th>Термин</th><th class="center">Встречается</th></tr></thead>
                        <tbody>{top_term_rows or '<tr><td colspan="2">Частотные термины отсутствуют.</td></tr>'}</tbody>
                    </table>
                </div>
                <div class="card">
                    <h3>TF-IDF</h3>
                    <table>
                        <thead><tr><th>Термин</th><th class="center">Средний TF-IDF</th></tr></thead>
                        <tbody>{tfidf_rows or '<tr><td colspan="2">TF-IDF признаки не рассчитаны.</td></tr>'}</tbody>
                    </table>
                </div>
            </div>
        </section>
        <section>
            <h2>Целевые термины</h2>
            {coverage_block or '<p>Целевые термины не заданы.</p>'}
        </section>
        {classification_section}
        <section>
            <h2>Группы дубликатов</h2>
            <p>Отражает потенциальные повторы документов и позволяет очистить корпус.</p>
            <table>
                <thead><tr><th>Каноничный документ</th><th class="center">Дубликаты</th><th>Примеры</th></tr></thead>
                <tbody>{duplicates_rows}</tbody>
            </table>
        </section>
        <section>
            <h2>Ошибки обработки (Top-10)</h2>
            <table>
                <thead><tr><th>Путь</th><th>Сообщение</th></tr></thead>
                <tbody>{error_rows}</tbody>
            </table>
        </section>
        <section>
            <h2>Итоги</h2>
            <ul class="summary-list">
                {summary_html}
            </ul>
        </section>
    </main>
    <footer>Файл создан автоматически анализатором analysis/eda.</footer>
</body>
</html>
"""

        dashboard_path.write_text(html_content, encoding="utf-8")

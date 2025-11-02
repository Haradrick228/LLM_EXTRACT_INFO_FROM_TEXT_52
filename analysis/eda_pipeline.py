from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.eda.config import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR
from analysis.eda.filesystem import collect_default_input_dirs, load_terms
from analysis.eda.models import PipelineSettings
from analysis.eda.pipeline import EDAPipeline

LOGGER = logging.getLogger("analysis.eda.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA-пайплайн для анализа и нормализации корпуса документов."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Каталоги с исходными документами (по умолчанию autodetect: scraped_site, gigachat_*).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Каталог для сохранения очищенных текстов (по умолчанию processed_eda_data).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Каталог для сохранения отчётов (по умолчанию analysis/reports).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Ограничение на количество обрабатываемых файлов.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать уже выгруженные очищенные тексты.",
    )
    parser.add_argument(
        "--terms-file",
        type=Path,
        default=None,
        help="Путь к текстовому файлу со списком целевых терминов (по одному на строку).",
    )
    parser.add_argument(
        "--no-output-text",
        action="store_true",
        help="Не сохранять очищенные тексты на диск.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логирования.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    root = Path.cwd()
    input_dirs = args.input_dirs or collect_default_input_dirs(root)
    if not input_dirs:
        raise ValueError("Не удалось обнаружить входные каталоги с документами.")

    output_dir = None if args.no_output_text else args.output_dir
    terms = load_terms(args.terms_file)

    settings = PipelineSettings(
        input_dirs=input_dirs,
        output_dir=output_dir,
        report_dir=args.report_dir,
        overwrite=args.overwrite,
        max_files=args.max_files,
        terms=terms,
    )
    pipeline = EDAPipeline(settings=settings)
    result = pipeline.run()

    print(f"JSON-отчёт: {result.json_report}")
    print(f"HTML-дашборд: {result.html_dashboard}")


if __name__ == "__main__":
    main()

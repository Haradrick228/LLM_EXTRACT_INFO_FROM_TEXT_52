"""Собрать человекочитаемый финальный отчёт по каноническому full-corpus прогону (Markdown + HTML)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.reporting.human_report_sections import build_html, build_markdown, load_bundle

DEFAULT_OUT = _ROOT / "eval" / "FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md"
DEFAULT_HTML = _ROOT / "analysis" / "reports" / "final_canonical_experiment_2026_05.html"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        type=Path,
        default=_ROOT / "eval" / "config" / "final_research_protocol.yaml",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--write-html",
        action="store_true",
        help="Записать analysis/reports/final_canonical_experiment_2026_05.html",
    )
    p.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    args = p.parse_args()

    bundle = load_bundle(_ROOT, args.protocol.resolve())
    report_md = build_markdown(bundle, _ROOT)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_md, encoding="utf-8")

    n_appendix = report_md.count("### ")
    n_metrics = len(bundle.metric_keys)
    print(
        f"Wrote {args.out.relative_to(_ROOT)} ({len(report_md)} chars, "
        f"appendix_tables={n_appendix}, metrics={n_metrics})"
    )

    if args.write_html:
        html_doc = build_html(bundle, _ROOT)
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_doc, encoding="utf-8")
        print(f"Wrote {args.html_out.relative_to(_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

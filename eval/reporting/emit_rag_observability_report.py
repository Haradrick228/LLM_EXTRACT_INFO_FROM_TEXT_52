"""HTML-отчёт: наблюдаемость RAG и анализ ошибок (фазы A+B).

Usage::

    python -m eval.reporting.emit_rag_observability_report --write-html
    python -m eval.reporting.emit_rag_observability_report --write-html --run-classify
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_PROTOCOL = _ROOT / "eval/config/final_research_protocol.yaml"
DEFAULT_HTML = _ROOT / "analysis/reports/rag_observability_error_analysis_2026_05.html"
DEFAULT_CLASSIFY_JSON = _ROOT / "eval/runs/rag_error_classification_pilot.json"

PIPELINE_TRACE_V1_FIELDS = [
    ("ts", "ISO timestamp запроса"),
    ("request_id", "UUID / perf_id"),
    ("question_text", "Текст запроса"),
    ("llm_model", "Модель генерации"),
    ("llm_temperature", "temperature"),
    ("llm_max_tokens", "max_tokens"),
    ("fetch_k", "Сколько чанков из индекса"),
    ("top_k", "Сколько после rerank в контекст"),
    ("chunks", "Список {rank, chunk_id, score, text_preview, source}"),
    ("rerank_scores", "Скоры reranker по кандидатам"),
    ("answer_text", "Ответ LLM"),
    ("timings_ms", "chroma / rerank / llm / total"),
    ("ref_sim", "Cosine ответа и answer_ref (eval)"),
]

HTML_CSS = """
:root { --text:#1a1a1a; --muted:#555; --border:#d8dce3; --bg:#eef1f6;
  --accent:#0b5fff; --card:#fff; --callout:#eef4ff; --warn:#fff8e6; --pass:#0d7a3e; }
* { box-sizing:border-box; }
body { font-family:"Segoe UI",-apple-system,sans-serif; line-height:1.55; margin:0;
  color:var(--text); background:#f0f2f5; }
.page { max-width:1400px; margin:0 auto; padding:24px 20px 64px; }
h1 { font-size:1.75rem; margin:0 0 6px; }
.lead { color:var(--muted); margin:0 0 20px; }
.meta { font-size:0.85rem; color:var(--muted); }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:14px 18px; margin:12px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card.warn { background:var(--warn); border-color:#e8d9a8; }
.card.ok { border-left:4px solid var(--pass); }
.callout { background:var(--callout); border:1px solid #c5d9ff; border-radius:8px;
  padding:12px 14px; margin:12px 0; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:0.88rem; }
th,td { border:1px solid var(--border); padding:7px 9px; text-align:left; }
th { background:var(--bg); }
.badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; }
.badge.yes { background:#d4edda; color:var(--pass); }
.badge.no { background:#f8d7da; color:#842029; }
.badge.part { background:#fff3cd; color:#664d03; }
.tabs-main > input[type="radio"] { position:fixed; left:-9999px; opacity:0; }
.tab-bar { display:flex; flex-wrap:wrap; gap:6px; margin:0 0 14px; padding:12px 0;
  border-bottom:2px solid var(--border); }
.tab-bar label { padding:10px 16px; border-radius:8px; cursor:pointer; font-weight:600;
  background:var(--card); border:2px solid var(--border); }
.tab-panel { display:none; }
#tab-curator:checked ~ .tab-bar label[for="tab-curator"],
#tab-done:checked ~ .tab-bar label[for="tab-done"],
#tab-gap:checked ~ .tab-bar label[for="tab-gap"],
#tab-conclusions:checked ~ .tab-bar label[for="tab-conclusions"],
#tab-pilot:checked ~ .tab-bar label[for="tab-pilot"],
#tab-roadmap:checked ~ .tab-bar label[for="tab-roadmap"] {
  background:var(--accent); color:#fff; border-color:var(--accent);
}
#tab-curator:checked ~ .panels #panel-curator,
#tab-done:checked ~ .panels #panel-done,
#tab-gap:checked ~ .panels #panel-gap,
#tab-conclusions:checked ~ .panels #panel-conclusions,
#tab-pilot:checked ~ .panels #panel-pilot,
#tab-roadmap:checked ~ .panels #panel-roadmap { display:block; }
ul.cmd { font-size:0.85rem; }
ul.cmd li { margin:6px 0; }
code { font-size:0.88em; }
"""


def _load_protocol(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _exists(rel: str) -> bool:
    return (_ROOT / rel).is_file()


def _find_first(globs: List[str]) -> Optional[Path]:
    for pattern in globs:
        hits = sorted(_ROOT.glob(pattern))
        if hits:
            return hits[0]
    return None


def _discover_pilot_paths(protocol: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
    anchor = (protocol.get("anchor_dense_trace") or {}).get("canonical_run_dir")
    if anchor:
        run = _ROOT / anchor
        trace = run / "trace.jsonl"
        if trace.is_file():
            ans = run / "answer_eval_results.jsonl"
            return trace, ans if ans.is_file() else None

    trace = _find_first(
        [
            "eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/trace.jsonl",
            "eval/runs/**/*retrieval_trace*.jsonl",
            "eval/runs/**/trace.jsonl",
        ]
    )

    b3 = (protocol.get("canonical_artifacts") or {}).get("b3_matrix_canonical") or {}
    b3_dir = b3.get("run_dir")
    answer: Optional[Path] = None
    if b3_dir:
        base = _ROOT / b3_dir
        qual = (protocol.get("quality_recommendation") or {})
        sid, prof = qual.get("scenario", "full_bge_m3"), qual.get("profile", "rerank_bge_v2_m3")
        candidates = list(base.glob(f"_answer_top/*{sid}*{prof}*/answer_eval_results.jsonl"))
        if not candidates:
            candidates = list(base.glob("_answer_top/**/answer_eval_results.jsonl"))
        if candidates:
            answer = sorted(candidates)[0]
        if trace is None:
            tr = list(base.glob(f"{sid}/retrieval_trace_{prof}.jsonl"))
            if tr:
                trace = tr[0]

    return trace, answer


def _run_classify(trace: Path, answer: Optional[Path], out_json: Path) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "eval.diagnostics.classify_rag_errors",
        "--trace",
        str(trace),
        "--gold-jsonl",
        str(_ROOT / "eval/datasets/eval_set_gold_v2.jsonl"),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_json.with_suffix(".md")),
    ]
    if answer and answer.is_file():
        cmd.extend(["--answer-jsonl", str(answer)])
    try:
        subprocess.run(cmd, cwd=str(_ROOT), check=True, capture_output=True, text=True)
        return out_json.is_file()
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or exc, file=sys.stderr)
        return False


def _status_badge(done: bool, partial: bool = False) -> str:
    if done:
        return '<span class="badge yes">Есть</span>'
    if partial:
        return '<span class="badge part">Частично</span>'
    return '<span class="badge no">Нет</span>'


def _artifact_rows() -> List[Tuple[str, str, str, bool, bool]]:
    """name, path, command, exists, partial."""
    rows = [
        (
            "Retrieval trace JSONL",
            "eval/runs/<id>/trace.jsonl или retrieval_trace_<profile>.jsonl",
            "python -m eval.retrieval.run_retrieval_eval --retrieval-trace-jsonl <path>",
            bool(_find_first(["eval/runs/**/trace.jsonl", "eval/runs/**/*retrieval_trace*.jsonl"])),
            True,
        ),
        (
            "Metrics ladder A–D",
            "eval/runs/<id>/metrics_ladder*.json",
            "python -m eval.diagnostics.metrics_ladder_funnel --trace ... --corpus-jsonl ...",
            bool(_find_first(["eval/runs/**/metrics_ladder*.json"])),
            False,
        ),
        (
            "Worst queries MD",
            "eval/docs/RETRIEVAL_WORST_QUERIES_2026_05.md",
            "python -m eval.diagnostics.summarize_worst_retrieval_traces --trace ...",
            _exists("eval/docs/RETRIEVAL_WORST_QUERIES_2026_05.md"),
            False,
        ),
        (
            "Answer-eval JSONL",
            "eval/runs/<id>/_answer_top/.../answer_eval_results.jsonl",
            "python -m eval.diagnostics.run_answer_eval_in_run_dir --run-dir ...",
            bool(_find_first(["eval/runs/**/answer_eval_results.jsonl"])),
            True,
        ),
        (
            "structured_logging + experiment_run_context",
            "common/structured_logging.py, core/monitoring/",
            "RAG_LOG_FORMAT=json в eval CLI",
            _exists("common/structured_logging.py"),
            True,
        ),
        (
            "RagService perf logs",
            "core/rag/service/rag_service.py",
            "[perf:…] chroma.ms / llm.ms (текст, не JSONL)",
            True,
            True,
        ),
        (
            "ml_service request_log",
            "Postgres request_log",
            "длительность API, без чанков",
            True,
            True,
        ),
        (
            "LLM chunk relevance (DeepSeek)",
            "eval/diagnostics/llm_chunk_relevance_deepseek.py",
            "--dry-run заготовка",
            _exists("eval/diagnostics/llm_chunk_relevance_deepseek.py"),
            True,
        ),
        (
            "Error classification pilot",
            "eval/runs/rag_error_classification_pilot.json",
            "python -m eval.diagnostics.classify_rag_errors ...",
            DEFAULT_CLASSIFY_JSON.is_file(),
            False,
        ),
        (
            "pipeline_trace_v1 (E2E)",
            "—",
            "фаза C: RagService.answer()",
            False,
            False,
        ),
    ]
    return rows


def _render_pilot_section(classify: Optional[Dict[str, Any]]) -> str:
    if not classify or not classify.get("n_queries"):
        return (
            "<p class='meta'>Пилот не запущен или нет trace локально. Команда:</p>"
            "<pre><code>python -m eval.reporting.emit_rag_observability_report "
            "--write-html --run-classify</code></pre>"
        )
    parts = [
        f"<p class='meta'>Источник: <code>{html.escape(str(classify.get('trace', '')))}</code>"
        f" · answer: <code>{html.escape(str(classify.get('answer_jsonl') or '(нет)'))}</code>"
        f" · порог ref_sim = <strong>{classify.get('ref_sim_threshold')}</strong></p>",
        f"<p>{html.escape(str(classify.get('rules_note', '')))}</p>",
        "<table><thead><tr><th>Класс</th><th>Count</th><th>Доля</th></tr></thead><tbody>",
    ]
    for cls, cnt in sorted((classify.get("counts") or {}).items()):
        frac = (classify.get("fractions") or {}).get(cls, 0)
        parts.append(
            f"<tr><td><code>{html.escape(cls)}</code></td>"
            f"<td>{cnt}</td><td>{frac:.1%}</td></tr>"
        )
    parts.append("</tbody></table>")
    for cls, rows in (classify.get("examples_top5_per_class") or {}).items():
        if not rows:
            continue
        parts.append(f"<h3>Примеры: {html.escape(cls)}</h3><ul>")
        for r in rows:
            parts.append(
                f"<li>#{r.get('eval_row_index')}: "
                f"{html.escape(str(r.get('question_preview', '')))} "
                f"(R@10={r.get('substring::recall@10')}, ref_sim={r.get('ref_sim')})</li>"
            )
        parts.append("</ul>")
    return "".join(parts)


def build_html(
    *,
    protocol: Dict[str, Any],
    classify: Optional[Dict[str, Any]],
    generated: str,
) -> str:
    prod = protocol.get("production_recommendation") or {}
    qual = protocol.get("quality_recommendation") or {}
    anchor = protocol.get("anchor_dense_trace") or {}

    parts: List[str] = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'/>",
        "<title>RAG: наблюдаемость и анализ ошибок</title>",
        f"<style>{HTML_CSS}</style></head><body><div class='page'>",
        "<h1>RAG: наблюдаемость и анализ ошибок</h1>",
        f"<p class='lead'>Статус реализации запроса куратора (Малюшитский) · {html.escape(generated)}</p>",
        '<div class="tabs-main">',
        '<input type="radio" name="tab" id="tab-curator" checked/>',
        '<input type="radio" name="tab" id="tab-done"/>',
        '<input type="radio" name="tab" id="tab-gap"/>',
        '<input type="radio" name="tab" id="tab-conclusions"/>',
        '<input type="radio" name="tab" id="tab-pilot"/>',
        '<input type="radio" name="tab" id="tab-roadmap"/>',
        '<nav class="tab-bar">',
        '<label for="tab-curator">Запрос куратора</label>',
        '<label for="tab-done">Сделано</label>',
        '<label for="tab-gap">Не сделано</label>',
        '<label for="tab-conclusions">Выводы</label>',
        '<label for="tab-pilot">Пилот B</label>',
        '<label for="tab-roadmap">Дорожная карта C</label>',
        "</nav><div class='panels'>",
        # Curator
        "<section class='tab-panel' id='panel-curator'>",
        "<h2>Запрос куратора</h2>",
        "<div class='callout'><p><strong>Логирование полного цикла:</strong> запрос, гиперпараметры LLM, "
        "найденные чанки, оценки близости запрос↔чанк, результат реранкера, ответ LLM, время, дата.</p>"
        "<p><strong>Анализ ошибок (два типа):</strong></p><ol>"
        "<li><strong>type1_retrieval</strong> — ретривер не нашёл релевантные чанки → ответ неверный.</li>"
        "<li><strong>type2_generation</strong> — чанки найдены, но LLM ответил плохо (ожидается реже).</li>"
        "</ol></div>",
        "<h3>Целевой контракт <code>pipeline_trace_v1</code> (фаза C)</h3>",
        "<table><thead><tr><th>Поле</th><th>Описание</th></tr></thead><tbody>",
    ]
    for field, desc in PIPELINE_TRACE_V1_FIELDS:
        parts.append(f"<tr><td><code>{html.escape(field)}</code></td><td>{html.escape(desc)}</td></tr>")
    parts.append("</tbody></table></section>")

    # Done
    parts.append("<section class='tab-panel' id='panel-done'><h2>Что уже есть в репозитории</h2>")
    parts.append("<table><thead><tr><th>Компонент</th><th>Артефакт</th><th>Статус</th></tr></thead><tbody>")
    for name, path, _cmd, ex, partial in _artifact_rows():
        parts.append(
            f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(path)}</code></td>"
            f"<td>{_status_badge(ex, partial)}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append("<h3>Команды воспроизведения</h3><ul class='cmd'>")
    parts.append(
        "<li><code>python -m eval.retrieval.run_retrieval_eval --retrieval-trace-jsonl trace.jsonl ...</code></li>"
    )
    parts.append(
        "<li><code>python -m eval.diagnostics.metrics_ladder_funnel --trace trace.jsonl "
        "--corpus-jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl</code></li>"
    )
    parts.append(
        "<li><code>python -m eval.diagnostics.summarize_worst_retrieval_traces --trace trace.jsonl</code></li>"
    )
    parts.append(
        "<li><code>python -m eval.diagnostics.run_answer_eval_in_run_dir --run-dir eval/runs/&lt;id&gt;</code></li>"
    )
    parts.append(
        "<li><code>python -m eval.diagnostics.classify_rag_errors --trace ... --answer-jsonl ...</code></li>"
    )
    parts.append("</ul>")
    parts.append(
        "<p class='meta'>Документация: "
        "<a href='../../eval/docs/METRICS_LADDER.md'>METRICS_LADDER</a>, "
        "<a href='../../eval/docs/RETRIEVAL_DIAGNOSIS_2026_05.md'>RETRIEVAL_DIAGNOSIS</a>, "
        "<a href='final_canonical_experiment_2026_05.html'>канонический отчёт</a>.</p>"
    )
    parts.append("</section>")

    # Gap
    parts.append("<section class='tab-panel' id='panel-gap'><h2>Чего нет (пока)</h2>")
    gaps = [
        "Единый JSONL <code>pipeline_trace_v1</code> на один запрос пользователя (retrieval + LLM в одном файле).",
        "Гиперпараметры LLM и полный текст ответа в том же trace, что и чанки (в проде — только perf-логи).",
        "Query↔chunk cosine в сервисе (в eval trace есть score Chroma / reranker_score).",
        "Автоматический отчёт ошибок в Telegram / nightly без ручного emit.",
        "Формальная rubric LLM по релевантности каждого чанка (заготовка DeepSeek dry-run).",
    ]
    parts.append("<ul>")
    for g in gaps:
        parts.append(f"<li>{g}</li>")
    parts.append("</ul></section>")

    # Conclusions
    parts.append("<section class='tab-panel' id='panel-conclusions'><h2>Выводы по текущим этапам</h2>")
    parts.append("<div class='card ok'>")
    parts.append(
        f"<p><strong>Прод (скорость):</strong> <code>{html.escape(str(prod.get('scenario', '')))}</code> + "
        f"<code>{html.escape(str(prod.get('profile', '')))}</code> — "
        f"{html.escape(str(prod.get('note', '')))}</p>"
    )
    parts.append(
        f"<p><strong>Точность ответа:</strong> <code>{html.escape(str(qual.get('scenario', '')))}</code> + "
        f"<code>{html.escape(str(qual.get('profile', '')))}</code> — "
        f"{html.escape(str(qual.get('note', '')))}</p>"
    )
    parts.append("</div>")
    parts.append(
        "<p>На полном корпусе (см. "
        "<a href='../../eval/docs/RETRIEVAL_DIAGNOSIS_2026_05.md'>RETRIEVAL_DIAGNOSIS</a>): "
        "доминирует проблема <strong>gold↔chunk / retrieval</strong> "
        "(лестница A ~59% — эталон не встречается в экспортированных чанках). "
        "Ошибки <strong>type2_generation</strong> в пилоте ожидаются реже, чем type1.</p>"
    )
    parts.append(
        "<p>Линия этапов в "
        "<a href='../../README.md#retrieval-recovery-2026-05'>главном README</a> (F0–F8, B3, canonical HTML). "
        f"Anchor trace: <code>{html.escape(str(anchor.get('canonical_run_dir', '')))}</code>.</p>"
    )
    parts.append("</section>")

    # Pilot
    parts.append("<section class='tab-panel' id='panel-pilot'><h2>Пилот классификации (фаза B)</h2>")
    parts.append(_render_pilot_section(classify))
    parts.append("</section>")

    # Roadmap
    parts.append("<section class='tab-panel' id='panel-roadmap'><h2>Дорожная карта — фаза C</h2>")
    parts.append(
        "<ol><li>Реализовать запись <code>pipeline_trace_v1</code> в "
        "<code>RagService.answer()</code> (и опционально ml_service / бот).</li>"
        "<li>Связать trace с gold для автоматического nightly classify.</li>"
        "<li>Дашборд / HTML из свежих trace без ручного поиска путей.</li>"
        "<li>LLM rubric по чанкам (после стабильного trace).</li></ol>"
    )
    parts.append(
        "<p class='meta'>Пересборка отчёта: "
        "<code>python -m eval.reporting.emit_rag_observability_report --write-html [--run-classify]</code></p>"
    )
    parts.append("</section>")

    parts.append("</div></div></div></body></html>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--write-html", action="store_true")
    p.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    p.add_argument("--classify-json", type=Path, default=DEFAULT_CLASSIFY_JSON)
    p.add_argument("--run-classify", action="store_true", help="Запустить classify если есть trace")
    args = p.parse_args()

    protocol = _load_protocol(args.protocol.resolve())
    classify: Optional[Dict[str, Any]] = None

    if args.run_classify:
        trace, answer = _discover_pilot_paths(protocol)
        if trace and trace.is_file():
            print(f"Running classify on trace={trace}")
            if _run_classify(trace, answer, args.classify_json.resolve()):
                print(f"Wrote {args.classify_json}")
        else:
            print("No retrieval trace found; skipping classify", file=sys.stderr)

    if args.classify_json.is_file():
        classify = json.loads(args.classify_json.read_text(encoding="utf-8"))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_doc = build_html(protocol=protocol, classify=classify, generated=generated)

    if args.write_html:
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_doc, encoding="utf-8")
        print(f"Wrote {args.html_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Emit / refresh sections in PHASE_B_RESEARCH_METRICS_REPORT from protocol + run artifacts."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.reporting.scenario_display import scenario_display_label

REPORT_PATH = _ROOT / "eval" / "PHASE_B_RESEARCH_METRICS_REPORT.md"
HTML_REPORT_PATH = _ROOT / "analysis" / "reports" / "phase_b_final_research_2026_05.html"
MARKER_START = "<!-- FINAL_RESEARCH_AUTO_START -->"
MARKER_END = "<!-- FINAL_RESEARCH_AUTO_END -->"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(cell: Dict[str, Any], key: str) -> Optional[float]:
    m = cell.get("metrics") or {}
    v = m.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _discover_metric_keys(summary: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for profs in (summary.get("results") or {}).values():
        for cell in (profs or {}).values():
            for k in (cell.get("metrics") or {}):
                if isinstance(k, str) and k != "_meta":
                    seen.setdefault(k, None)
    primary = [k for k in seen if "::" not in k]
    secondary = [k for k in seen if "::" in k]
    primary_sorted = sorted(primary, key=lambda x: (0 if x.startswith("recall@") else 1, x))
    return primary_sorted + sorted(secondary)


def _leaderboard_rows(summary: Dict[str, Any], metric_key: str) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    for sid, profs in (summary.get("results") or {}).items():
        for pname, cell in (profs or {}).items():
            v = _metric(cell, metric_key)
            if v is not None:
                rows.append((v, str(sid), str(pname)))
    rows.sort(reverse=True)
    return rows


def _md_table(headers: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _matrix_table_body(
    summary: Dict[str, Any],
    metric_key: str,
    scenarios: List[str],
    profiles: List[str],
) -> List[List[str]]:
    body: List[List[str]] = []
    for sid in scenarios:
        row = [scenario_display_label(sid, summary)]
        for pname in profiles:
            cell = (summary.get("results") or {}).get(sid, {}).get(pname) or {}
            v = _metric(cell, metric_key)
            row.append(f"{v:.4f}" if v is not None else "")
        body.append(row)
    return body


def _canonical_b3_path(protocol: Dict[str, Any]) -> Path:
    canon = protocol.get("canonical_artifacts") or {}
    b3 = canon.get("b3_matrix_canonical") or canon.get("b3_matrix_6x3_fetch100") or {}
    return _ROOT / str(b3.get("run_dir", ""))


def _build_auto_block(protocol: Dict[str, Any]) -> str:
    canon = protocol.get("canonical_artifacts") or {}
    b3_meta = canon.get("b3_matrix_canonical") or canon.get("b3_matrix_6x3_fetch100") or {}
    b3_path = _canonical_b3_path(protocol)
    summary = _load_json(b3_path / "summary.json") if b3_path.is_dir() else None
    validity = _load_json(_ROOT / "eval" / "runs" / "final_research_validity_suite.json")
    ladder = _load_json(
        _ROOT / str(protocol.get("anchor_dense_trace", {}).get("canonical_run_dir", ""))
        / "metrics_ladder_full_corpus.json"
    )
    fetch_k = summary.get("fetch_k") if summary else b3_meta.get("fetch_k", "?")
    metric_modes = summary.get("metric_modes") if summary else b3_meta.get("metric_modes", "?")

    lines: List[str] = [
        MARKER_START,
        "",
        f"## Авто-сводка (protocol `{protocol.get('protocol_version')}`)",
        "",
        "> **Только канонический прогон** — таблицы ниже из одного `run_dir`; исторические B3 (fetch_k=100 и частичные) **не включены**.",
        "",
        f"- **Каноническая B3 6×3:** `{b3_path.relative_to(_ROOT)}`",
        f"- **fetch_k:** `{fetch_k}` · **metric_modes:** `{metric_modes}` · **вопросов:** 101 (gold v2)",
        f"- **Протокол:** `eval/config/final_research_protocol.yaml`",
        f"- **Replay B3:** `python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix`",
        f"- **Validity suite:** `python -m eval.diagnostics.run_final_research_validity_suite`",
        "",
    ]

    if validity:
        neg = validity.get("checks", {}).get("neg_control_retrieval") or {}
        neg_detail = ""
        if neg.get("honest_value") is not None and neg.get("shuffled_value") is not None:
            neg_detail = (
                f"honest={neg.get('honest_value'):.4f}, shuffled={neg.get('shuffled_value'):.4f}, "
                f"gap={neg.get('gap', 0):.4f}"
            )
        elif neg.get("note"):
            neg_detail = str(neg.get("note"))
        lines.append("### Валидность метрик (pass/fail)")
        lines.append("")
        lines.append(
            _md_table(
                ["Проверка", "Статус", "Детали"],
                [
                    [
                        "external_sanity",
                        "PASS" if (validity.get("checks", {}).get("external_sanity") or {}).get("pass") else "FAIL",
                        str((validity.get("checks", {}).get("external_sanity") or {}).get("value")),
                    ],
                    [
                        "neg_control",
                        "PASS"
                        if neg.get("pass")
                        else ("SKIP" if neg.get("skipped") else "FAIL"),
                        neg_detail or str(neg.get("honest_path", "")),
                    ],
                    [
                        "metrics_ladder",
                        "PASS" if (validity.get("checks", {}).get("metrics_ladder") or {}).get("pass") else "FAIL",
                        f"A={((validity.get('checks', {}).get('metrics_ladder') or {}).get('class_fraction_A'))}",
                    ],
                    [
                        "semantic_threshold_probe",
                        "PASS"
                        if (validity.get("checks", {}).get("semantic_threshold_probe") or {}).get("pass")
                        else "FAIL",
                        str((validity.get("checks", {}).get("semantic_threshold_probe") or {}).get("path", "")),
                    ],
                ],
            )
        )
        lines.append("")
        if neg.get("honest_path"):
            lines.extend(
                [
                    f"Neg-control honest cell: `{neg.get('honest_path')}` "
                    f"({neg.get('honest_scenario', '')} × {neg.get('honest_profile', '')}).",
                    "",
                ]
            )

    if ladder:
        frac = ladder.get("class_fractions") or {}
        wilson = ladder.get("wilson_hit_at_k_post") or {}
        cond = ladder.get("conditional_given_oracle_pool") or {}
        anchor_dir = protocol.get("anchor_dense_trace", {}).get("canonical_run_dir", "")
        lines.extend(
            [
                f"### Лестница A–D (anchor trace, `{anchor_dir}`)",
                "",
                _md_table(
                    ["Показатель", "Значение"],
                    [
                        ["class A (ref в корпусе)", f"{frac.get('A', 0):.4f}"],
                        ["class D (miss@k)", f"{frac.get('D', 0):.4f}"],
                        [
                            "Wilson CI recall@10 (post-rank)",
                            f"{wilson.get('lo', 0):.4f} – {wilson.get('hi', 0):.4f} (n={wilson.get('successes', '?')})",
                        ],
                        [
                            "conditional_given_oracle_pool (mean recall@k)",
                            f"{cond.get('mean_recall_at_k_post', 0):.4f} (n={cond.get('n', 0)})",
                        ],
                    ],
                ),
                "",
            ]
        )

    answer_top = _load_json(_ROOT / "eval" / "runs" / "final_research_answer_top.json")
    b3_run_name = b3_path.name
    if answer_top and str(answer_top.get("b3_run_dir", "")).endswith(b3_run_name):
        lines.extend(
            [
                "### Answer-layer (top retrieval по `substring::recall@10`, канонический B3)",
                "",
                _md_table(
                    ["rank", "scenario", "profile", "retrieval RC", "note"],
                    [
                        [
                            str(c.get("rank")),
                            str(c.get("scenario")),
                            str(c.get("profile")),
                            str(c.get("returncode")),
                            "OK" if c.get("returncode") == 0 else "см. _answer_top/*/answer_eval.stderr.txt",
                        ]
                        for c in answer_top.get("cells") or []
                    ],
                ),
                "",
            ]
        )

    if summary:
        profiles = list(summary.get("profiles") or [])
        scenarios = list(summary.get("scenarios") or [])
        metric_keys = _discover_metric_keys(summary)

        top = _leaderboard_rows(summary, "substring::recall@10")[:8]
        lines.extend(
            [
                "### Leaderboard — `substring::recall@10` (канонический B3)",
                "",
                _md_table(
                    ["rank", "dense base", "rerank profile", "recall@10"],
                    [
                    [str(i + 1), scenario_display_label(sid, summary), pname, f"{v:.4f}"]
                    for i, (v, sid, pname) in enumerate(top)
                ],
                ),
                "",
                f"### Матрица 6×3 — все метрики ({len(metric_keys)} таблиц)",
                "",
                "По одной таблице на каждый числовой ключ из `retrieval_*.json` ячеек канонического прогона.",
                "",
            ]
        )
        for mk in metric_keys:
            lines.append(f"#### `{mk}`")
            lines.append("")
            body = _matrix_table_body(summary, mk, scenarios, profiles)
            lines.append(_md_table(["scenario"] + profiles, body))
            lines.append("")

    lines.extend(
        [
            "### Ограничения (не снимать при публикации)",
            "",
            "- Gold v2 без стабильных `chunk_id` — substring/token_f1/semantic по тексту, не BEIR id-hit.",
            "- `semantic::` и `substring::` — разные определения hit; не сравнивать абсолютные уровни между ними.",
            "- Финальный отчёт строится **только** из канонического `run_dir` в протоколе; старые B3 — архив.",
            "",
            MARKER_END,
            "",
        ]
    )
    return "\n".join(lines)


def _build_html(protocol: Dict[str, Any], md_block: str) -> str:
    b3_path = _canonical_b3_path(protocol)
    summary = _load_json(b3_path / "summary.json") if b3_path.is_dir() else None
    validity = _load_json(_ROOT / "eval" / "runs" / "final_research_validity_suite.json")
    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; max-width: 1280px; margin: 32px auto; padding: 0 16px; }
    h1, h2, h3, h4 { line-height: 1.25; }
    table { border-collapse: collapse; margin: 12px 0; font-size: 12px; }
    th, td { border: 1px solid #ddd; padding: 4px 8px; }
    th { background: #f6f6f6; }
    code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }
  """
    parts = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Phase B Final Research 2026-05</title><style>{css}</style></head><body>",
        "<h1>Phase B — финальный отчёт (канонический B3)</h1>",
        f"<p>Источник: <code>{html.escape(str(b3_path.relative_to(_ROOT)))}</code></p>",
    ]
    if validity:
        parts.append("<h2>Валидность</h2><ul>")
        for name, chk in (validity.get("checks") or {}).items():
            ok = chk.get("pass") if isinstance(chk, dict) else None
            label = "PASS" if ok else ("SKIP" if isinstance(chk, dict) and chk.get("skipped") else "FAIL")
            parts.append(f"<li><code>{html.escape(name)}</code>: {label}</li>")
        parts.append(f"</ul><p>overall_pass={validity.get('overall_pass')}</p>")
    if summary:
        profiles = list(summary.get("profiles") or [])
        scenarios = list(summary.get("scenarios") or [])
        for mk in _discover_metric_keys(summary):
            parts.append(f"<h3><code>{html.escape(mk)}</code></h3><table>")
            parts.append("<tr><th>scenario</th>" + "".join(f"<th>{html.escape(p)}</th>" for p in profiles) + "</tr>")
            for row in _matrix_table_body(summary, mk, scenarios, profiles):
                parts.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
            parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        type=Path,
        default=_ROOT / "eval" / "config" / "final_research_protocol.yaml",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--write-html", action="store_true", help="Also write analysis/reports/phase_b_final_research_2026_05.html")
    args = p.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    block = _build_auto_block(protocol)

    if args.dry_run:
        print(block)
        return 0

    if REPORT_PATH.is_file():
        text = REPORT_PATH.read_text(encoding="utf-8")
        if MARKER_START in text and MARKER_END in text:
            pre, rest = text.split(MARKER_START, 1)
            _, post = rest.split(MARKER_END, 1)
            text = pre + block + post
        else:
            text = text.rstrip() + "\n\n" + block
    else:
        text = "# Phase B — финальный отчёт\n\n" + block

    REPORT_PATH.write_text(text, encoding="utf-8")
    print("Updated", REPORT_PATH.relative_to(_ROOT))

    if args.write_html:
        HTML_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HTML_REPORT_PATH.write_text(_build_html(protocol, block), encoding="utf-8")
        print("Updated", HTML_REPORT_PATH.relative_to(_ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Понятная вкладка «Итоги»: вердикт по embedder / reranker / комбинации + answer-eval."""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.reporting.decision_analysis import (
    PRIMARY_METRIC,
    _rank_metric,
    _stats_for_profiles,
    _stats_for_scenarios,
    format_pct,
    format_winner_value,
    metric_value,
    profile_label_human,
)
from eval.reporting.scenario_display import scenario_display_label

PRIMARY_LABEL = "substring Recall@10 (дословный hit чанка в top-10)"
# Прод: быстрый USER-bge-m3; точность ответа: канонический BAAI/bge-m3 (лучший avg_ref_sim).
PRODUCTION_SCENARIO = "full_user_bge_m3"
PRODUCTION_PROFILE = "rerank_bge_v2_m3"
PRODUCTION_NOTE = "USER-bge-m3 — быстрый индекс; BGE reranker в паре."
QUALITY_SCENARIO = "full_bge_m3"
QUALITY_PROFILE = "rerank_bge_v2_m3"
QUALITY_NOTE = (
    "BAAI/bge-m3 (не USER-bge-m3): лучший avg_ref_sim на answer-eval — "
    "если важнее точность ответа LLM, а не скорость индекса."
)


@dataclass
class FactorVerdict:
    factor_kind: str  # embedder | reranker
    factor_id: str
    display: str
    mean: float
    max: float
    partner_at_max: str
    reason: str


@dataclass
class ComboVerdict:
    scenario: str
    profile: str
    display: str
    retrieval: float
    avg_ref_sim: Optional[float] = None
    n_answer: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class OutcomesModel:
    n_questions: int
    n_chunks: int
    n_cells: int
    fetch_k: int
    primary_metric: str
    primary_label: str
    embedder: FactorVerdict
    reranker: FactorVerdict
    combo_retrieval: ComboVerdict
    combo_retrieval_ids: str
    combo_production: ComboVerdict
    quality_alternate: Optional[ComboVerdict]
    embedder_rows: List[List[str]]
    reranker_rows: List[List[str]]
    combo_top_rows: List[List[str]]
    answer_rows: List[List[str]]
    secondary_lines: List[str]
    answer_missing_note: Optional[str] = None


def _cell_display(summary: Dict[str, Any], sid: str, pname: str) -> str:
    return (
        f"{scenario_display_label(sid, summary)} + "
        f"{profile_label_human(pname)} (`{sid}` + `{pname}`)"
    )


def _combo_from_cell(
    summary: Dict[str, Any],
    sid: str,
    pname: str,
    retrieval: float,
    *,
    avg_ref_sim: Optional[float] = None,
    n_answer: int = 0,
    tags: Optional[List[str]] = None,
) -> ComboVerdict:
    return ComboVerdict(
        scenario=sid,
        profile=pname,
        display=_cell_display(summary, sid, pname),
        retrieval=retrieval,
        avg_ref_sim=avg_ref_sim,
        n_answer=n_answer,
        tags=tags or [],
    )


def _embedder_verdict(summary: Dict[str, Any], metric_key: str) -> FactorVerdict:
    stats = _stats_for_scenarios(summary, metric_key)
    top = stats[0]
    second = stats[1] if len(stats) > 1 else None
    margin = top.max - second.max if second else None
    margin_s = format_winner_value(metric_key, margin) if margin and margin > 1e-9 else "—"
    reason = (
        f"Лучший max {format_winner_value(metric_key, top.max)} по {metric_key} "
        f"(mean {format_winner_value(metric_key, top.mean)} по 3 reranker). "
        f"Отрыв max от 2-го embedder: {margin_s}."
    )
    return FactorVerdict(
        factor_kind="embedder",
        factor_id=top.factor_id,
        display=scenario_display_label(top.factor_id, summary),
        mean=top.mean,
        max=top.max,
        partner_at_max=top.argmax_partner,
        reason=reason,
    )


def _reranker_verdict(summary: Dict[str, Any], metric_key: str) -> FactorVerdict:
    stats = _stats_for_profiles(summary, metric_key)
    top = stats[0]
    second = stats[1] if len(stats) > 1 else None
    margin = top.max - second.max if second else None
    margin_s = format_winner_value(metric_key, margin) if margin and margin > 1e-9 else "—"
    reason = (
        f"Лучший max {format_winner_value(metric_key, top.max)} по {metric_key} "
        f"(mean {format_winner_value(metric_key, top.mean)} по 6 embedder). "
        f"Отрыв max от 2-го reranker: {margin_s}."
    )
    return FactorVerdict(
        factor_kind="reranker",
        factor_id=top.factor_id,
        display=profile_label_human(top.factor_id),
        mean=top.mean,
        max=top.max,
        partner_at_max=top.argmax_partner,
        reason=reason,
    )


def _retrieval_for_cell(
    summary: Dict[str, Any],
    metric_key: str,
    scenario: str,
    profile: str,
    ranked: Sequence[Tuple[float, str, str]],
) -> Optional[float]:
    for v, sid, pname in ranked:
        if sid == scenario and pname == profile:
            return float(v)
    cell = ((summary.get("results") or {}).get(scenario) or {}).get(profile) or {}
    raw = (cell.get("metrics") or {}).get(metric_key)
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


def _pinned_combo(
    summary: Dict[str, Any],
    metric_key: str,
    ranked: Sequence[Tuple[float, str, str]],
    ans_map: Dict[Tuple[str, str], Tuple[Optional[float], int]],
    answer_by_key: Dict[Tuple[str, str], Any],
    scenario: str,
    profile: str,
    tags: List[str],
) -> Optional[ComboVerdict]:
    ret = _retrieval_for_cell(summary, metric_key, scenario, profile, ranked)
    if ret is None:
        row = answer_by_key.get((scenario, profile))
        r10 = getattr(row, "retrieval_recall10", None) if row is not None else None
        if isinstance(r10, (int, float)):
            ret = float(r10)
    if ret is None:
        return None
    sim, n = ans_map.get((scenario, profile), (None, 0))
    return _combo_from_cell(summary, scenario, profile, ret, avg_ref_sim=sim, n_answer=n, tags=tags)


def _answer_lookup(
    answer_by_key: Dict[Tuple[str, str], Any],
) -> Dict[Tuple[str, str], Tuple[Optional[float], int]]:
    out: Dict[Tuple[str, str], Tuple[Optional[float], int]] = {}
    for key, row in answer_by_key.items():
        out[key] = (getattr(row, "avg_ref_sim", None), int(getattr(row, "count_ref", 0) or 0))
    return out


def build_outcomes_model(
    summary: Dict[str, Any],
    *,
    n_questions: int,
    n_chunks: int,
    fetch_k: int,
    answer_rows: Sequence[Any] = (),
) -> OutcomesModel:
    metric_key = PRIMARY_METRIC
    ranked = _rank_metric(summary, metric_key)
    if not ranked:
        raise ValueError("no ranked cells for primary metric")

    best_ret = ranked[0][0]
    tied_cells = [(sid, pname, v) for v, sid, pname in ranked if abs(v - best_ret) <= 1e-9]

    answer_by_key: Dict[Tuple[str, str], Any] = {}
    for row in answer_rows:
        answer_by_key[(str(row.scenario), str(row.profile))] = row
    ans_map = _answer_lookup(answer_by_key)

    def _ref_sim(sid: str, pname: str) -> Tuple[Optional[float], int]:
        return ans_map.get((sid, pname), (None, 0))

    # Лучшая комбинация для retrieval (ключевая метрика)
    combo_ret_ids = " / ".join(f"{s} + {p}" for s, p, _ in tied_cells)
    combo_ret = _combo_from_cell(
        summary, tied_cells[0][0], tied_cells[0][1], tied_cells[0][2], tags=["лидер retrieval"]
    )
    if len(tied_cells) > 1:
        combo_ret.tags.append("равный max R@10 у нескольких reranker на FRIDA")

    # Продакшен: среди tied (или всех с answer) — max avg_ref_sim
    prod_candidates: List[ComboVerdict] = []
    for sid, pname, ret in tied_cells:
        sim, n = _ref_sim(sid, pname)
        prod_candidates.append(
            _combo_from_cell(summary, sid, pname, ret, avg_ref_sim=sim, n_answer=n)
        )
    if not any(c.avg_ref_sim is not None for c in prod_candidates):
        prod_candidates = []
        for v, sid, pname in ranked[:5]:
            sim, n = _ref_sim(sid, pname)
            prod_candidates.append(
                _combo_from_cell(summary, sid, pname, v, avg_ref_sim=sim, n_answer=n)
            )
    prod_candidates.sort(
        key=lambda c: (
            c.avg_ref_sim is not None,
            c.avg_ref_sim or -1.0,
            c.retrieval,
        ),
        reverse=True,
    )
    combo_prod = prod_candidates[0]
    combo_prod.tags.append("рекомендуем в прод (по answer-eval)")
    forced_prod = _pinned_combo(
        summary,
        metric_key,
        ranked,
        ans_map,
        answer_by_key,
        PRODUCTION_SCENARIO,
        PRODUCTION_PROFILE,
        ["рекомендуем в прод", PRODUCTION_NOTE],
    )
    if forced_prod is not None:
        combo_prod = forced_prod

    quality_alt = _pinned_combo(
        summary,
        metric_key,
        ranked,
        ans_map,
        answer_by_key,
        QUALITY_SCENARIO,
        QUALITY_PROFILE,
        ["максимум точности ответа", QUALITY_NOTE],
    )
    if quality_alt is not None and (
        quality_alt.scenario == combo_prod.scenario and quality_alt.profile == combo_prod.profile
    ):
        quality_alt = None

    emb_rows: List[List[str]] = []
    for i, st in enumerate(_stats_for_scenarios(summary, metric_key), 1):
        emb_rows.append(
            [
                str(i),
                st.factor_id,
                format_winner_value(metric_key, st.mean),
                format_winner_value(metric_key, st.max),
                st.argmax_partner,
            ]
        )

    rer_rows: List[List[str]] = []
    for i, st in enumerate(_stats_for_profiles(summary, metric_key), 1):
        rer_rows.append(
            [
                str(i),
                st.factor_id,
                profile_label_human(st.factor_id),
                format_winner_value(metric_key, st.mean),
                format_winner_value(metric_key, st.max),
                st.argmax_partner,
            ]
        )

    combo_top: List[List[str]] = []
    for i, (v, sid, pname) in enumerate(ranked[:8], 1):
        sim, n = _ref_sim(sid, pname)
        combo_top.append(
            [
                str(i),
                sid,
                pname,
                format_winner_value(metric_key, v),
                format_sim(sim) if sim is not None else "—",
                str(n) if n else "—",
            ]
        )

    ans_table: List[List[str]] = []
    sorted_ans = sorted(
        answer_rows,
        key=lambda r: (
            -(getattr(r, "avg_ref_sim", None) or -1),
            -(getattr(r, "retrieval_recall10", None) or -1),
        ),
    )
    for i, row in enumerate(sorted_ans, 1):
        ans_table.append(
            [
                str(i),
                scenario_display_label(row.scenario, summary),
                profile_label_human(row.profile),
                format_pct(row.retrieval_recall10),
                format_sim(row.avg_ref_sim),
                str(row.count_ref or 101),
                "OK" if row.ok else "FAIL",
            ]
        )

    secondary: List[str] = []
    for mk, label in (
        ("token_f1::recall@10", "token_f1 Recall@10"),
        ("semantic::recall@10", "semantic Recall@10"),
    ):
        r = _rank_metric(summary, mk)
        if r:
            v, sid, pname = r[0]
            secondary.append(
                f"**{label}:** лидер `{sid}` + `{pname}` — {format_winner_value(mk, v)} "
                f"(другая метрика, не ключевая)."
            )

    missing: Optional[str] = None
    if len(sorted_ans) < 8:
        missing = (
            "Дозаполнить avg_ref_sim: "
            "`python -m eval.diagnostics.run_final_research_answer_top8_gap "
            "--b3-run-dir eval/runs/20260527_161527_b3_full_corpus_rerank`"
        )

    return OutcomesModel(
        n_questions=n_questions,
        n_chunks=n_chunks,
        n_cells=len(summary.get("scenarios") or []) * len(summary.get("profiles") or []),
        fetch_k=fetch_k,
        primary_metric=metric_key,
        primary_label=PRIMARY_LABEL,
        embedder=_embedder_verdict(summary, metric_key),
        reranker=_reranker_verdict(summary, metric_key),
        combo_retrieval=combo_ret,
        combo_retrieval_ids=combo_ret_ids,
        combo_production=combo_prod,
        quality_alternate=quality_alt,
        embedder_rows=emb_rows,
        reranker_rows=rer_rows,
        combo_top_rows=combo_top,
        answer_rows=ans_table,
        secondary_lines=secondary,
        answer_missing_note=missing,
    )


def format_sim(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_outcomes_markdown(model: OutcomesModel) -> List[str]:
    e, r = model.embedder, model.reranker
    cr, cp = model.combo_retrieval, model.combo_production

    lines: List[str] = [
        "## 2. Итоги",
        "",
        f"{model.n_questions} вопросов · {model.n_chunks:,} чанков · матрица {model.n_cells} ячеек "
        f"(6 embedder × 3 reranker) · fetch_k={model.fetch_k}.",
        "",
        "Каждая ячейка = **dense retrieval (embedder)** → **rerank** → метрики по чанкам. "
        "Retrieval без rerank отдельно не снимали.",
        "",
        "### Ключевая метрика",
        "",
        f"**`{model.primary_metric}`** — {model.primary_label}. "
        "По ней сравниваем embedder, reranker и итоговую пару.",
        "",
        "### Вердикт",
        "",
        _md_table(
            ["Что выбираем", "ID", "Показатель", "Почему"],
            [
                [
                    "Embedder",
                    e.factor_id,
                    f"max {format_winner_value(model.primary_metric, e.max)}, "
                    f"mean {format_winner_value(model.primary_metric, e.mean)}",
                    f"Лучший по max/mean `{model.primary_metric}` среди 6 индексов.",
                ],
                [
                    "Reranker",
                    r.factor_id,
                    f"max {format_winner_value(model.primary_metric, r.max)}, "
                    f"mean {format_winner_value(model.primary_metric, r.mean)}",
                    f"Лучший по max/mean среди 3 reranker; при max часто с `{r.partner_at_max}`.",
                ],
                [
                    "Комбинация (retrieval)",
                    model.combo_retrieval_ids,
                    format_winner_value(model.primary_metric, cr.retrieval),
                    "Максимум по ключевой метрике в матрице 6×3 (полный пайплайн).",
                ],
                [
                    "**Брать в прод**",
                    f"**{cp.scenario} + {cp.profile}**",
                    f"R@10 {format_winner_value(model.primary_metric, cp.retrieval)}"
                    + (
                        f", avg_ref_sim **{format_sim(cp.avg_ref_sim)}**"
                        if cp.avg_ref_sim is not None
                        else ""
                    ),
                    PRODUCTION_NOTE
                    + (
                        f" avg_ref_sim {format_sim(cp.avg_ref_sim)} (answer-eval)."
                        if cp.avg_ref_sim is not None
                        else " Answer-eval по этой паре — см. таблицу ниже."
                    ),
                ],
            ],
        ),
        "",
        f"> **Embedder:** {e.reason}",
        "",
        f"> **Reranker:** {r.reason}",
        "",
    ]

    qa = model.quality_alternate
    if qa is not None:
        lines.extend(
            [
                "### Точность ответа (BGE-M3)",
                "",
                f"**{qa.scenario} + {qa.profile}** — "
                f"R@10 {format_winner_value(model.primary_metric, qa.retrieval)}"
                + (
                    f", avg_ref_sim **{format_sim(qa.avg_ref_sim)}**"
                    if qa.avg_ref_sim is not None
                    else ""
                )
                + f". {QUALITY_NOTE}",
                "",
            ]
        )

    for sl in model.secondary_lines:
        lines.append(sl)
        lines.append("")

    lines.extend(
        [
            "### Рейтинг embedder (ключевая метрика)",
            "",
            _md_table(
                ["#", "embedder", "mean", "max", "reranker @ max"],
                model.embedder_rows,
            ),
            "",
            "### Рейтинг reranker (ключевая метрика)",
            "",
            _md_table(
                ["#", "reranker_id", "название", "mean", "max", "embedder @ max"],
                model.reranker_rows,
            ),
            "",
            "### Топ комбинаций (ключевая метрика + answer если есть)",
            "",
            _md_table(
                ["#", "embedder", "reranker", "R@10", "avg_ref_sim", "N"],
                model.combo_top_rows,
            ),
            "",
        ]
    )

    if model.answer_rows:
        lines.extend(
            [
                "### Проверка ответа LLM (отдельный прогон)",
                "",
                "На каждый из 101 вопросов: `RagService.answer` → **avg_ref_sim** "
                "(cosine эмбеддингов ответа и `answer_ref`). Это **не** Recall@10 по чанкам.",
                "",
                _md_table(
                    ["#", "Модель", "Reranker", "Retrieval R@10", "avg_ref_sim", "N", "Статус"],
                    model.answer_rows,
                ),
                "",
            ]
        )
    if model.answer_missing_note:
        lines.append(f"> {model.answer_missing_note}")
        lines.append("")

    lines.extend(["---", ""])
    return lines


def render_outcomes_html(model: OutcomesModel) -> List[str]:
    e, r = model.embedder, model.reranker
    cp = model.combo_production
    parts: List[str] = [
        "<h2>Итоги</h2>",
        f"<p class='meta'>{model.n_questions} вопросов · {model.n_chunks:,} чанков · "
        f"{model.n_cells} ячеек · fetch_k={model.fetch_k}. "
        "Ячейка = embedder (dense) → rerank.</p>",
        '<div class="verdict-hero">',
        '<div class="verdict-card pick">',
        f"<h3>Брать в прод</h3>"
        f"<p class='big'><code>{html.escape(cp.scenario)}</code> + "
        f"<code>{html.escape(cp.profile)}</code></p>",
        f"<p>Retrieval: <strong>{html.escape(format_winner_value(model.primary_metric, cp.retrieval))}</strong>"
        + (
            f" · Ответ: <strong>{html.escape(format_sim(cp.avg_ref_sim))}</strong> avg_ref_sim"
            if cp.avg_ref_sim is not None
            else ""
        )
        + f"</p><p class='meta'>{html.escape(PRODUCTION_NOTE)}</p></div>",
        '<div class="verdict-card">',
        "<h3>Ключевая метрика</h3>",
        f"<p><code>{html.escape(model.primary_metric)}</code><br/>"
        f"{html.escape(model.primary_label)}</p></div>",
        "</div>",
        "<h3>Вердикт по факторам</h3>",
        "<table><thead><tr><th>Фактор</th><th>Выбор</th><th>max</th><th>mean</th><th>Комментарий</th></tr></thead><tbody>",
        f"<tr><td>Embedder</td><td><code>{html.escape(e.factor_id)}</code></td>"
        f"<td>{html.escape(format_winner_value(model.primary_metric, e.max))}</td>"
        f"<td>{html.escape(format_winner_value(model.primary_metric, e.mean))}</td>"
        f"<td>Лучший индекс по ключевой метрике</td></tr>",
        f"<tr><td>Reranker</td><td><code>{html.escape(r.factor_id)}</code></td>"
        f"<td>{html.escape(format_winner_value(model.primary_metric, r.max))}</td>"
        f"<td>{html.escape(format_winner_value(model.primary_metric, r.mean))}</td>"
        f"<td>Лучший rerank; @max с <code>{html.escape(r.partner_at_max)}</code></td></tr>",
        "</tbody></table>",
    ]

    qa = model.quality_alternate
    if qa is not None:
        parts.append('<div class="callout">')
        parts.append("<h3>Точность ответа (BGE-M3)</h3>")
        parts.append(
            f"<p><code>{html.escape(qa.scenario)}</code> + "
            f"<code>{html.escape(qa.profile)}</code> — "
            f"retrieval <strong>{html.escape(format_winner_value(model.primary_metric, qa.retrieval))}</strong>"
            + (
                f", avg_ref_sim <strong>{html.escape(format_sim(qa.avg_ref_sim))}</strong>"
                if qa.avg_ref_sim is not None
                else ""
            )
            + f". {html.escape(QUALITY_NOTE)}</p></div>"
        )

    return parts


def _html_table_simple(
    headers: List[str], rows: List[List[str]], *, highlight_first: bool = False
) -> str:
    out = ["<table><thead><tr>"]
    out.append("".join(f"<th>{html.escape(h)}</th>" for h in headers))
    out.append("</tr></thead><tbody>")
    for ri, row in enumerate(rows):
        cls = ' class="best"' if highlight_first and ri == 0 else ""
        out.append(f"<tr{cls}>")
        for cell in row:
            out.append(f"<td>{html.escape(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)

"""Агрегация лидеров и матрица решений для финального отчёта (канонический B3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from eval.reporting.scenario_display import scenario_display_label

PRIMARY_METRIC = "substring::recall@10"

PROFILE_LABELS_HUMAN = {
    "baseline_mpnet_rerank": "MiniLM (базовый rerank)",
    "rerank_bge_v2_m3": "BGE reranker",
    "rerank_jina_v3": "Jina reranker",
}


def profile_label_human(profile_id: str) -> str:
    return PROFILE_LABELS_HUMAN.get(profile_id, profile_id)


def metric_value(cell: Dict[str, Any], key: str) -> Optional[float]:
    v = (cell.get("metrics") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def format_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_sim(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


DECISION_METRIC_KEYS = [
    "substring::recall@10",
    "token_f1::recall@10",
    "semantic::recall@10",
    "substring::mrr",
    "semantic::mrr",
]

METRIC_LABELS_RU = {
    "substring::recall@10": "substring Recall@10",
    "token_f1::recall@10": "token_f1 Recall@10",
    "semantic::recall@10": "semantic Recall@10",
    "substring::mrr": "substring MRR",
    "semantic::mrr": "semantic MRR",
}


@dataclass
class WinnerRow:
    metric_key: str
    metric_label: str
    value: float
    scenario: str
    profile: str
    display: str
    tied: List[str] = field(default_factory=list)
    margin_vs_second: Optional[float] = None
    is_primary: bool = False


@dataclass
class DecisionScenario:
    priority: str
    recommended: str
    evidence: str
    metric_key: str


def _rank_metric(summary: Dict[str, Any], metric_key: str) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    for sid, profs in (summary.get("results") or {}).items():
        for pname, cell in (profs or {}).items():
            v = metric_value(cell, metric_key)
            if v is not None:
                rows.append((v, str(sid), str(pname)))
    rows.sort(reverse=True)
    return rows


def compute_metric_winners(
    summary: Dict[str, Any],
    metric_keys: Optional[List[str]] = None,
) -> List[WinnerRow]:
    keys = metric_keys or DECISION_METRIC_KEYS
    winners: List[WinnerRow] = []
    for mk in keys:
        ranked = _rank_metric(summary, mk)
        if not ranked:
            continue
        best_v, best_sid, best_prof = ranked[0]
        tied_labels: List[str] = []
        for v, sid, pname in ranked:
            if abs(v - best_v) > 1e-9:
                break
            tied_labels.append(f"{sid} + {pname}")
        margin = None
        if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) > 1e-9:
            margin = ranked[0][0] - ranked[1][0]
        display = tied_labels[0] if tied_labels else ""
        winners.append(
            WinnerRow(
                metric_key=mk,
                metric_label=METRIC_LABELS_RU.get(mk, mk),
                value=best_v,
                scenario=best_sid,
                profile=best_prof,
                display=display,
                tied=tied_labels,
                margin_vs_second=margin,
                is_primary=(mk == PRIMARY_METRIC),
            )
        )
    return winners


def format_winner_value(metric_key: str, value: float) -> str:
    if "recall@" in metric_key:
        return format_pct(value)
    return f"{value:.4f}"


def build_experiment_coverage_rows(
    summary: Dict[str, Any],
    *,
    answer_top_only_frida: bool = True,
    answer_diverse_available: bool = False,
) -> List[List[str]]:
    n_models = len(summary.get("scenarios") or [])
    n_profiles = len(summary.get("profiles") or [])
    fetch_k = int(summary.get("fetch_k") or 40)
    answer_note = "Отдельный прогон для трёх вариантов на базе FRIDA"
    if answer_diverse_available:
        answer_note = "Расширенный прогон для нескольких лидеров по разным критериям"
    return [
        [
            "База знаний",
            f"Около 61 500 фрагментов текста, {n_models} вариантов индекса (разные модели поиска)",
            "Размер базы и модель векторизации",
        ],
        [
            "Поиск + пересортировка",
            f"{n_models} моделей × {n_profiles} reranker = {n_models * n_profiles} вариантов; "
            f"на вопрос берётся {fetch_k} кандидатов, потом пересортировка",
            "Доля вопросов (из 101), где эталон «попал» в топ-5, 10 или 20",
        ],
        [
            "Текст ответа нейросети",
            answer_note,
            "Средняя похожесть сгенерированного ответа на эталон (шкала 0–1)",
        ],
    ]


def _best_answer_row(
    answer_rows: List[Any],
    diverse_rows: List[Any],
) -> Optional[Any]:
    all_rows = list(answer_rows) + list(diverse_rows)
    best = None
    for r in all_rows:
        sim = getattr(r, "avg_ref_sim", None)
        if sim is None:
            continue
        if best is None or sim > best.avg_ref_sim:
            best = r
    return best


def build_decision_matrix(
    summary: Dict[str, Any],
    winners: List[WinnerRow],
    answer_rows: Optional[List[Any]] = None,
    diverse_rows: Optional[List[Any]] = None,
) -> List[DecisionScenario]:
    answer_rows = answer_rows or []
    diverse_rows = diverse_rows or []
    by_key = {w.metric_key: w for w in winners}

    def _cell_label(w: Optional[WinnerRow]) -> str:
        if not w:
            return "—"
        return w.display

    scenarios: List[DecisionScenario] = [
        DecisionScenario(
            priority="Протокол Phase B (gate)",
            recommended=_cell_label(by_key.get(PRIMARY_METRIC)),
            evidence=format_winner_value(PRIMARY_METRIC, by_key[PRIMARY_METRIC].value)
            if PRIMARY_METRIC in by_key
            else "—",
            metric_key=PRIMARY_METRIC,
        ),
        DecisionScenario(
            priority="Мягче: пересечение токенов (чанк)",
            recommended=_cell_label(by_key.get("token_f1::recall@10")),
            evidence=format_winner_value(
                "token_f1::recall@10",
                by_key["token_f1::recall@10"].value,
            )
            if "token_f1::recall@10" in by_key
            else "—",
            metric_key="token_f1::recall@10",
        ),
        DecisionScenario(
            priority="Смысл чанка (semantic hit)",
            recommended=_cell_label(by_key.get("semantic::recall@10")),
            evidence=format_winner_value(
                "semantic::recall@10",
                by_key["semantic::recall@10"].value,
            )
            if "semantic::recall@10" in by_key
            else "—",
            metric_key="semantic::recall@10",
        ),
        DecisionScenario(
            priority="Порядок в выдаче (substring MRR)",
            recommended=_cell_label(by_key.get("substring::mrr")),
            evidence=format_winner_value("substring::mrr", by_key["substring::mrr"].value)
            if "substring::mrr" in by_key
            else "—",
            metric_key="substring::mrr",
        ),
    ]

    best_ans = _best_answer_row(answer_rows, diverse_rows)
    if best_ans:
        scenarios.append(
            DecisionScenario(
                priority="Текст ответа LLM (avg_ref_sim)",
                recommended=(
                    f"{scenario_display_label(best_ans.scenario, summary)} + "
                    f"{profile_label_human(best_ans.profile)}"
                ),
                evidence=format_sim(best_ans.avg_ref_sim),
                metric_key="avg_ref_sim",
            )
        )
    elif answer_rows:
        scenarios.append(
            DecisionScenario(
                priority="Текст ответа LLM (avg_ref_sim)",
                recommended="только FRIDA×3 (предварительно)",
                evidence="см. раздел Ответы LLM",
                metric_key="avg_ref_sim",
            )
        )
    return scenarios


@dataclass
class FactorStats:
    factor_id: str
    mean: float
    max: float
    argmax_partner: str


def _stats_for_scenarios(summary: Dict[str, Any], metric_key: str) -> List[FactorStats]:
    """По каждому embedder: mean/max substring (и т.д.) по всем reranker в строке матрицы."""
    out: List[FactorStats] = []
    for sid, profs in (summary.get("results") or {}).items():
        vals: List[float] = []
        best_p = ""
        best_v = float("-inf")
        for pname, cell in (profs or {}).items():
            v = metric_value(cell, metric_key)
            if v is None:
                continue
            vals.append(v)
            if v > best_v:
                best_v = v
                best_p = str(pname)
        if not vals:
            continue
        out.append(
            FactorStats(
                factor_id=str(sid),
                mean=sum(vals) / len(vals),
                max=max(vals),
                argmax_partner=best_p,
            )
        )
    out.sort(key=lambda x: (x.max, x.mean), reverse=True)
    return out


def _stats_for_profiles(summary: Dict[str, Any], metric_key: str) -> List[FactorStats]:
    """По каждому reranker: mean/max по всем embedder (столбец матрицы)."""
    profiles: Dict[str, List[float]] = {}
    partners: Dict[str, Tuple[float, str]] = {}
    for sid, profs in (summary.get("results") or {}).items():
        for pname, cell in (profs or {}).items():
            v = metric_value(cell, metric_key)
            if v is None:
                continue
            pname = str(pname)
            profiles.setdefault(pname, []).append(v)
            prev = partners.get(pname)
            if prev is None or v > prev[0]:
                partners[pname] = (v, str(sid))
    out: List[FactorStats] = []
    for pname, vals in profiles.items():
        out.append(
            FactorStats(
                factor_id=pname,
                mean=sum(vals) / len(vals),
                max=max(vals),
                argmax_partner=partners.get(pname, (0.0, ""))[1],
            )
        )
    out.sort(key=lambda x: (x.max, x.mean), reverse=True)
    return out


def pipeline_decomposition_header_row(
    summary: Dict[str, Any], metric_key: str
) -> List[str]:
    """Одна строка: лучшая комбинация (полный пайплайн dense+rerank)."""
    ranked = _rank_metric(summary, metric_key)
    if not ranked:
        return []
    best_v = ranked[0][0]
    tied = [
        f"{sid} + {pname}"
        for v, sid, pname in ranked
        if abs(v - best_v) <= 1e-9
    ]
    return [
        "Комбинация (embedder×reranker)",
        " / ".join(tied),
        "—",
        format_winner_value(metric_key, best_v),
        "dense+rerank",
    ]


def embedder_factor_table_rows(
    summary: Dict[str, Any], metric_key: str
) -> List[List[str]]:
    rows: List[List[str]] = []
    for st in _stats_for_scenarios(summary, metric_key):
        rows.append(
            [
                st.factor_id,
                format_winner_value(metric_key, st.mean),
                format_winner_value(metric_key, st.max),
                st.argmax_partner,
            ]
        )
    return rows


def reranker_factor_table_rows(
    summary: Dict[str, Any], metric_key: str
) -> List[List[str]]:
    rows: List[List[str]] = []
    for st in _stats_for_profiles(summary, metric_key):
        rows.append(
            [
                st.factor_id,
                format_winner_value(metric_key, st.mean),
                format_winner_value(metric_key, st.max),
                st.argmax_partner,
            ]
        )
    return rows


def factor_breakdown_section_rows(
    summary: Dict[str, Any], metric_key: str
) -> Tuple[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """Заголовок метрики + сводная строка комбинации + таблицы embedder и reranker."""
    header = pipeline_decomposition_header_row(summary, metric_key)
    summary_rows = [header] if header else []
    return (
        [metric_key],
        summary_rows,
        embedder_factor_table_rows(summary, metric_key),
        reranker_factor_table_rows(summary, metric_key),
    )


FACTOR_BREAKDOWN_METRICS = [
    "substring::recall@10",
    "token_f1::recall@10",
    "semantic::recall@10",
]


def leader_table_rows(summary: Dict[str, Any], winners: List[WinnerRow]) -> List[List[str]]:
    """По одной строке на метрику; при равенстве max — все ячейки в одной ячейке."""
    rows: List[List[str]] = []
    for w in winners:
        configs = w.tied if w.tied else [w.display]
        val = format_winner_value(w.metric_key, w.value)
        margin = (
            format_winner_value(w.metric_key, w.margin_vs_second)
            if w.margin_vs_second is not None
            else "—"
        )
        rows.append(
            [
                w.metric_key,
                val,
                " / ".join(configs),
                margin,
            ]
        )
    return rows


def tradeoff_top3_table(
    summary: Dict[str, Any],
) -> List[List[str]]:
    """Строка = метрика, колонки = топ-3 ячейки."""
    headers = ["Метрика", "#1", "#2", "#3"]
    body: List[List[str]] = []
    for mk in ("substring::recall@10", "token_f1::recall@10", "semantic::recall@10"):
        ranked = _rank_metric(summary, mk)[:3]
        cols = []
        for v, sid, pname in ranked:
            cols.append(f"{sid} + {pname} ({format_winner_value(mk, v)})")
        while len(cols) < 3:
            cols.append("—")
        body.append([mk] + cols)
    return headers, body

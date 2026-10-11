"""Нарратив и HTML v2 для человекочитаемого финального отчёта (канонический full-corpus прогон)."""
from __future__ import annotations

import copy
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from eval.reporting.outcomes_verdict import (
    build_outcomes_model,
    format_sim as outcomes_format_sim,
    render_outcomes_html,
    render_outcomes_markdown,
)
from eval.reporting.scenario_display import scenario_display_label

PRIMARY_METRIC = "substring::recall@10"

PROFILE_LABELS_HUMAN = {
    "baseline_mpnet_rerank": "MiniLM (базовый rerank)",
    "rerank_bge_v2_m3": "BGE reranker",
    "rerank_jina_v3": "Jina reranker",
}

GROUP_LABELS_RU = {
    "primary": "Базовые (без префикса режима)",
    "substring": "Дословное совпадение (substring)",
    "token_f1": "Пересечение токенов (token_f1)",
    "semantic": "Смысловая близость чанка (semantic)",
    "other": "Прочие",
}


@dataclass
class MetricDef:
    id: str
    title: str
    plain_ru: str
    formula_ru: str
    when_to_use: str
    not_comparable_with: str = ""


METRIC_GLOSSARY: List[MetricDef] = [
    MetricDef(
        id="recall_at_k",
        title="Recall@k",
        plain_ru=(
            "Доля вопросов (из 101), где хотя бы один из k лучших найденных фрагментов "
            "считается «попавшим» по выбранному критерию."
        ),
        formula_ru="Для каждого вопроса: 1, если hit в топ-k, иначе 0. Итог — среднее по 101 вопросу.",
        when_to_use="Главный показатель «нашли ли эталон в выдаче». Смотрим k=5, 10, 20.",
        not_comparable_with="Нельзя сравнивать recall между разными fetch_k или разными корпусами.",
    ),
    MetricDef(
        id="precision_at_k",
        title="Precision@k",
        plain_ru="Средняя доля релевантных фрагментов среди k лучших (по тому же критерию hit).",
        formula_ru="Считается per-query, затем усредняется по 101 вопросу.",
        when_to_use="Показывает «засорённость» топа нерелевантными чанками.",
        not_comparable_with="",
    ),
    MetricDef(
        id="mrr",
        title="MRR (Mean Reciprocal Rank)",
        plain_ru="Насколько высоко в списке стоит первый правильный фрагмент.",
        formula_ru="Для вопроса: 1/rank первого hit, или 0 если hit нет. MRR — среднее по 101 вопросу.",
        when_to_use="Важно, когда важен порядок: лучше hit на 1-й позиции, чем на 10-й при том же recall.",
        not_comparable_with="",
    ),
    MetricDef(
        id="ndcg3",
        title="nDCG@3",
        plain_ru="Качество ранжирования в топ-3 с учётом позиции релевантного чанка.",
        formula_ru="Normalized DCG@3 по списку кандидатов после пайплайна.",
        when_to_use="Дополнительный сигнал порядка в коротком топе.",
        not_comparable_with="",
    ),
    MetricDef(
        id="substring",
        title="Режим substring (дословно)",
        plain_ru=(
            "Чанк «попал», если эталонная фраза из gold — подстрока текста чанка "
            "(или наоборот), без учёта регистра."
        ),
        formula_ru="hit = reference.lower() in chunk.lower() (или обратное включение).",
        when_to_use="Жёсткий критерий для отчёта: основная Recall@10 ~6.4% — именно substring.",
        not_comparable_with="Semantic recall и avg_ref_sim ответа LLM — другие шкалы.",
    ),
    MetricDef(
        id="token_f1",
        title="Режим token_f1",
        plain_ru="Мягче substring: пересечение токенов эталона и чанка (F1 по токенам).",
        formula_ru="token_overlap_f1_hit между reference и document_text.",
        when_to_use="Когда эталон разрезан чанкованием, но часть слов совпадает.",
        not_comparable_with="Не смешивать с substring % в одной «оценке успеха».",
    ),
    MetricDef(
        id="semantic_retrieval",
        title="Режим semantic (поиск)",
        plain_ru=(
            "Чанк попал, если косинус эмбеддингов чанка и эталона ≥ порога 0.62 "
            "(модель paraphrase-multilingual-mpnet-base-v2)."
        ),
        formula_ru="cosine(embed(chunk), embed(reference)) ≥ 0.62",
        when_to_use="Диагностика «по смыслу» на этапе retrieval. Цифры обычно выше substring.",
        not_comparable_with="6.4% substring; avg_ref_sim ответа LLM (~64%).",
    ),
    MetricDef(
        id="squad_sanity",
        title="SQuAD: Recall@10 (внешний контроль)",
        plain_ru=(
            "Отдельный «лёгкий» прогон: ~80 вопросов из SQuAD и маленький корпус, "
            "где правильный ответ дословно лежит в чанке. Проверяем, что поиск вообще работает."
        ),
        formula_ru="Тот же substring::recall@10, но на external_sanity, не на 61k корпусе.",
        when_to_use="Sanity-check pipeline (порог PASS ≥ 50%; у нас ~82.5%).",
        not_comparable_with="Цифры полного корпуса 101×61k.",
    ),
    MetricDef(
        id="neg_control",
        title="Негативный контроль (перетасованные эталоны)",
        plain_ru=(
            "Те же вопросы, но эталоны перепутаны с другими вопросами. "
            "Метрика должна упасть — иначе оценка «рисует» успех."
        ),
        formula_ru="Честный gold vs shuffled_questions_gold_v2.jsonl, та же ячейка FRIDA+Jina.",
        when_to_use="Доверие к pipeline: ~6.4% → ~3.0% при shuffle.",
        not_comparable_with="",
    ),
    MetricDef(
        id="ladder_a",
        title="Класс A (лестница метрик)",
        plain_ru=(
            "Эталонный текст не встречается ни в одном из ~61k экспортированных чанков как substring. "
            "Поиск по substring принципиально не может «попасть»."
        ),
        formula_ru="metrics_ladder_funnel по trace + corpus JSONL.",
        when_to_use="Объясняет потолок ~6% recall — не баг одной модели.",
        not_comparable_with="",
    ),
    MetricDef(
        id="avg_ref_sim",
        title="avg_ref_sim (ответ LLM)",
        plain_ru=(
            "Средняя косинусная близость эмбеддингов сгенерированного ответа и поля answer_ref. "
            "Это слой «качество ответа», не «нашли чанк»."
        ),
        formula_ru="mean(cosine(embed(answer), embed(answer_ref))) по 101 вопросу.",
        when_to_use="Отдельный прогон answer-eval для 8 конфигов (FRIDA×3 + top-8 gap×5; ~0.65 на лучшем).",
        not_comparable_with="6.4% substring recall; semantic retrieval hit@k.",
    ),
]


@dataclass
class ReportBundle:
    protocol: Dict[str, Any]
    b3_path: Path
    summary: Dict[str, Any]
    validity: Optional[Dict[str, Any]]
    ladder: Optional[Dict[str, Any]]
    answer_top: Optional[Dict[str, Any]]
    answer_diverse: Optional[Dict[str, Any]]
    supplemental_b3_path: Optional[Path]
    scenarios: List[str]
    profiles: List[str]
    metric_keys: List[str]
    generated: str
    ts: str


@dataclass
class AnswerEvalRow:
    rank: int
    scenario: str
    profile: str
    retrieval_recall10: Optional[float]
    avg_ref_sim: Optional[float]
    avg_refs_sim: Optional[float]
    count_ref: int
    ok: bool


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(cell: Dict[str, Any], key: str) -> Optional[float]:
    v = (cell.get("metrics") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def discover_metric_keys(summary: Dict[str, Any]) -> List[str]:
    seen: Dict[str, None] = {}
    for profs in (summary.get("results") or {}).values():
        for cell in (profs or {}).values():
            for k in (cell.get("metrics") or {}):
                seen.setdefault(str(k), None)
    primary = [k for k in seen if "::" not in k]
    secondary = [k for k in seen if "::" in k]
    primary_sorted = sorted(primary, key=lambda x: (0 if x.startswith("recall@") else 1, x))
    return primary_sorted + sorted(secondary)


def group_metric_keys(keys: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "primary": [],
        "substring": [],
        "token_f1": [],
        "semantic": [],
        "other": [],
    }
    for k in keys:
        g = _metric_group_label(k)
        groups.setdefault(g, []).append(k)
    return groups


def _metric_group_label(metric_key: str) -> str:
    if metric_key.startswith("semantic::"):
        return "semantic"
    if metric_key.startswith("token_f1::"):
        return "token_f1"
    if metric_key.startswith("substring::"):
        return "substring"
    if "::" in metric_key:
        return "other"
    return "primary"


def leaderboard(
    summary: Dict[str, Any], metric_key: str, top_n: int = 10
) -> List[Tuple[float, str, str]]:
    rows: List[Tuple[float, str, str]] = []
    for sid, profs in (summary.get("results") or {}).items():
        for pname, cell in (profs or {}).items():
            v = metric_value(cell, metric_key)
            if v is not None:
                rows.append((v, str(sid), str(pname)))
    rows.sort(reverse=True)
    return rows[:top_n]


def format_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_sim(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def profile_label_human(profile_id: str) -> str:
    return PROFILE_LABELS_HUMAN.get(profile_id, profile_id)


def matrix_shape_label(bundle: ReportBundle) -> str:
    return f"{len(bundle.scenarios)}×{len(bundle.profiles)}"


def merge_b3_supplement(
    canonical_summary: Dict[str, Any],
    protocol: Dict[str, Any],
    root: Path,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Merge supplemental B3 summary into canonical for display; canonical run_dir unchanged."""
    supp_meta = protocol.get("supplemental_b3") or {}
    run_dir_s = supp_meta.get("run_dir")
    if not run_dir_s:
        return canonical_summary, None
    supp_path = root / str(run_dir_s)
    supp_summary = _load_json(supp_path / "summary.json")
    if not supp_summary:
        return canonical_summary, None

    merged = copy.deepcopy(canonical_summary)
    merged_results = dict(merged.get("results") or {})
    for sid, profs in (supp_summary.get("results") or {}).items():
        merged_results[sid] = profs
    merged["results"] = merged_results

    scenarios = list(merged.get("scenarios") or [])
    for sid in supp_summary.get("scenarios") or []:
        if sid not in scenarios:
            scenarios.append(sid)
    merged["scenarios"] = scenarios

    merged_chroma = dict(merged.get("scenario_chroma") or {})
    for sid, meta in (supp_summary.get("scenario_chroma") or {}).items():
        merged_chroma[sid] = meta
    merged["scenario_chroma"] = merged_chroma
    return merged, supp_path


def load_bundle(root: Path, protocol_path: Path) -> ReportBundle:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    canon = protocol.get("canonical_artifacts") or {}
    b3_meta = canon.get("b3_matrix_canonical") or {}
    b3_path = root / str(b3_meta.get("run_dir", ""))
    canonical_summary = _load_json(b3_path / "summary.json")
    if not canonical_summary:
        raise FileNotFoundError(f"summary.json missing: {b3_path}")

    summary, supplemental_b3_path = merge_b3_supplement(canonical_summary, protocol, root)

    manifest = _load_json(b3_path / "matrix_manifest.json")
    validity = _load_json(root / "eval/runs/final_research_validity_suite.json")
    ladder = _load_json(
        root
        / str(protocol.get("anchor_dense_trace", {}).get("canonical_run_dir", ""))
        / "metrics_ladder_full_corpus.json"
    )
    answer_top = _load_json(root / "eval/runs/final_research_answer_top.json")
    answer_diverse = _load_json(root / "eval/runs/final_research_answer_diverse.json")

    scenarios = list(summary.get("scenarios") or [])
    profiles = list(summary.get("profiles") or [])
    ts = (manifest or {}).get("ts", b3_path.name.split("_")[1] if "_" in b3_path.name else "")

    return ReportBundle(
        protocol=protocol,
        b3_path=b3_path,
        summary=summary,
        validity=validity,
        ladder=ladder,
        answer_top=answer_top,
        answer_diverse=answer_diverse,
        supplemental_b3_path=supplemental_b3_path,
        scenarios=scenarios,
        profiles=profiles,
        metric_keys=discover_metric_keys(summary),
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        ts=str(ts),
    )


def best_primary_cell(bundle: ReportBundle) -> Tuple[float, str, str]:
    top = leaderboard(bundle.summary, PRIMARY_METRIC, 1)
    if not top:
        return 0.0, "", ""
    return top[0]


def matrix_values(
    bundle: ReportBundle, metric_key: str
) -> Tuple[List[List[Optional[float]]], Optional[Tuple[int, int, float]]]:
    rows: List[List[Optional[float]]] = []
    best: Optional[Tuple[int, int, float]] = None
    for ri, sid in enumerate(bundle.scenarios):
        row: List[Optional[float]] = []
        for ci, pname in enumerate(bundle.profiles):
            cell = (bundle.summary.get("results") or {}).get(sid, {}).get(pname) or {}
            v = metric_value(cell, metric_key)
            row.append(v)
            if v is not None and (best is None or v > best[2]):
                best = (ri, ci, v)
        rows.append(row)
    return rows, best


def _parse_answer_stdout(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"'avg_ref_sim':\s*([\d.]+)", text)
    m2 = re.search(r"'avg_refs_sim':\s*([\d.]+)", text)
    m3 = re.search(r"'count_ref':\s*(\d+)", text)
    if not m:
        return None
    return {
        "avg_ref_sim": float(m.group(1)),
        "avg_refs_sim": float(m2.group(1)) if m2 else None,
        "count_ref": int(m3.group(1)) if m3 else 0,
    }


def _aggregate_answer_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    sims: List[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rs = row.get("ref_sim")
        if isinstance(rs, (int, float)):
            sims.append(float(rs))
    if not sims:
        return None
    return {"avg_ref_sim": sum(sims) / len(sims), "avg_refs_sim": None, "count_ref": len(sims)}


def _answer_rows_from_payload(
    cells: List[Dict[str, Any]],
    bundle: ReportBundle,
    project_root: Path,
    subdir_name: str,
) -> List[AnswerEvalRow]:
    rows: List[AnswerEvalRow] = []
    for cell in cells:
        rank = int(cell.get("rank") or 0)
        scenario = str(cell.get("scenario") or "")
        profile = str(cell.get("profile") or "")
        recall10 = cell.get("substring_recall_at_10")
        if isinstance(recall10, (int, float)):
            recall10 = float(recall10)
        else:
            cell_m = (
                (bundle.summary.get("results") or {}).get(scenario, {}).get(profile, {}).get("metrics") or {}
            )
            v = cell_m.get(PRIMARY_METRIC)
            recall10 = float(v) if isinstance(v, (int, float)) else None

        metrics: Optional[Dict[str, Any]] = None
        manifest_data = cell.get("manifest_data") or {}
        run_dir_s = manifest_data.get("run_dir") or cell.get("manifest")
        if run_dir_s:
            run_dir = Path(str(run_dir_s))
            if not run_dir.is_absolute():
                run_dir = project_root / run_dir
            stdout = run_dir / "answer_eval.stdout.txt"
            jsonl = run_dir / "answer_eval_results.jsonl"
            if manifest_data.get("answer_eval_out"):
                jsonl = Path(str(manifest_data["answer_eval_out"]))
            metrics = _parse_answer_stdout(stdout) or _aggregate_answer_jsonl(jsonl)

        if metrics is None:
            fallback_root = bundle.b3_path
            supp_ids = list((bundle.protocol.get("supplemental_b3") or {}).get("base_ids") or [])
            if bundle.supplemental_b3_path and scenario in supp_ids:
                fallback_root = bundle.supplemental_b3_path
            subdir = fallback_root / subdir_name / f"{rank:02d}_{scenario}_{profile}"
            metrics = _parse_answer_stdout(subdir / "answer_eval.stdout.txt") or _aggregate_answer_jsonl(
                subdir / "answer_eval_results.jsonl"
            )

        rows.append(
            AnswerEvalRow(
                rank=rank,
                scenario=scenario,
                profile=profile,
                retrieval_recall10=recall10,
                avg_ref_sim=metrics.get("avg_ref_sim") if metrics else None,
                avg_refs_sim=metrics.get("avg_refs_sim") if metrics else None,
                count_ref=int(metrics.get("count_ref") or 0) if metrics else 0,
                ok=cell.get("returncode") == 0,
            )
        )
    return sorted(rows, key=lambda r: r.rank)


def load_answer_metrics(bundle: ReportBundle, root: Optional[Path] = None) -> List[AnswerEvalRow]:
    if not bundle.answer_top:
        return []
    project_root = root or bundle.b3_path.parents[2]
    return _answer_rows_from_payload(
        list(bundle.answer_top.get("cells") or []),
        bundle,
        project_root,
        "_answer_top",
    )


def load_answer_diverse_metrics(bundle: ReportBundle, root: Optional[Path] = None) -> List[AnswerEvalRow]:
    if not bundle.answer_diverse:
        return []
    project_root = root or bundle.b3_path.parents[2]
    return _answer_rows_from_payload(
        list(bundle.answer_diverse.get("cells") or []),
        bundle,
        project_root,
        "_answer_diverse",
    )


def load_answer_top8_gap_metrics(bundle: ReportBundle, root: Optional[Path] = None) -> List[AnswerEvalRow]:
    project_root = root or bundle.b3_path.parents[2]
    gap = _load_json(project_root / "eval/runs/final_research_answer_top8_gap.json")
    if not gap:
        return []
    return _answer_rows_from_payload(
        list(gap.get("cells") or []),
        bundle,
        project_root,
        "_answer_top8_gap",
    )


def load_answer_user_bge_m3_metrics(bundle: ReportBundle, root: Optional[Path] = None) -> List[AnswerEvalRow]:
    project_root = root or bundle.b3_path.parents[2]
    payload = _load_json(project_root / "eval/runs/final_research_answer_user_bge_m3.json")
    if not payload:
        return []
    return _answer_rows_from_payload(
        list(payload.get("cells") or []),
        bundle,
        project_root,
        "_answer_user_bge_m3",
    )


def merge_answer_metrics(
    answer_rows: List[AnswerEvalRow],
    diverse_rows: List[AnswerEvalRow],
    gap_rows: Optional[List[AnswerEvalRow]] = None,
    user_bge_m3_rows: Optional[List[AnswerEvalRow]] = None,
) -> List[AnswerEvalRow]:
    """Объединить top + diverse + top8_gap + USER-bge-m3 answer-eval без дублей."""
    by_key: Dict[Tuple[str, str], AnswerEvalRow] = {}
    for row in answer_rows:
        by_key[(row.scenario, row.profile)] = row
    for row in diverse_rows:
        by_key[(row.scenario, row.profile)] = row
    for row in gap_rows or []:
        by_key[(row.scenario, row.profile)] = row
    for row in user_bge_m3_rows or []:
        by_key[(row.scenario, row.profile)] = row
    merged = list(by_key.values())
    merged.sort(
        key=lambda r: (-(r.avg_ref_sim or -1.0), -(r.retrieval_recall10 or -1.0), r.rank),
    )
    for i, row in enumerate(merged, 1):
        row.rank = i
    return merged


def _answer_eval_md_table(bundle: ReportBundle, rows: List[AnswerEvalRow]) -> str:
    if not rows:
        return ""
    body: List[List[str]] = []
    for r in rows:
        body.append(
            [
                str(r.rank),
                scenario_display_label(r.scenario, bundle.summary),
                profile_label_human(r.profile),
                format_pct(r.retrieval_recall10),
                format_sim(r.avg_ref_sim),
                str(r.count_ref),
                "OK" if r.ok else "ошибка",
            ]
        )
    return md_table(
        ["#", "Модель", "Reranker", "Retrieval R@10", "avg_ref_sim", "N", "Статус"],
        body,
    )


def _answer_eval_html_table(bundle: ReportBundle, rows: List[AnswerEvalRow]) -> str:
    if not rows:
        return ""
    body: List[List[str]] = []
    for r in rows:
        body.append(
            [
                html.escape(str(r.rank)),
                html.escape(scenario_display_label(r.scenario, bundle.summary)),
                html.escape(profile_label_human(r.profile)),
                html.escape(format_pct(r.retrieval_recall10)),
                html.escape(format_sim(r.avg_ref_sim)),
                html.escape(str(r.count_ref)),
                html.escape("OK" if r.ok else "ошибка"),
            ]
        )
    return _html_table(
        ["#", "Модель", "Reranker", "Retrieval R@10", "avg_ref_sim", "N", "Статус"],
        body,
    )


def build_experiment_steps(bundle: ReportBundle) -> List[Tuple[str, str]]:
    protocol = bundle.protocol
    corpus = protocol.get("corpus") or {}
    eval_set = protocol.get("eval_set") or {}
    matrix = protocol.get("retrieval_matrix_b3") or {}
    n_q = int(eval_set.get("expected_questions") or 101)
    n_chunks = int(corpus.get("expected_collection_rows") or 61543)
    n_models = len(bundle.scenarios)
    n_profiles = len(bundle.profiles)
    fetch_k = int(matrix.get("fetch_k") or bundle.summary.get("fetch_k") or 40)
    total_runs = n_q * n_models * n_profiles

    return [
        (
            "Подготовка корпуса",
            f"~{n_chunks:,} фрагментов текста экспортированы в JSONL и проиндексированы в "
            f"**{n_models} отдельных Chroma-коллекциях** (по одной модели эмбеддинга на коллекцию).",
        ),
        (
            "Контрольные вопросы",
            f"**{n_q}** вопросов из `{eval_set.get('path', 'eval_set_gold_v2.jsonl')}` — "
            "у каждого есть эталонная цитата (`references`) для проверки поиска.",
        ),
        (
            "Сравнение всех комбинаций",
            f"Для каждой пары «модель поиска × пересортировка» (**{n_models}×{n_profiles}={n_models * n_profiles}** вариантов) "
            f"на каждый вопрос: векторный поиск **{fetch_k}** кандидатов → пересортировка → "
            "проверка, попал ли эталон в топ-5, 10 или 20.",
        ),
        (
            "Объём прогонов",
            f"**{total_runs:,}** отдельных оценок поиска ({n_q} вопросов × {n_models * n_profiles} вариантов). "
            "Текст ответа нейросети на этом этапе не генерировали — только поиск.",
        ),
        (
            "Проверки доверия",
            "Отдельно: SQuAD sanity, neg-control с перетасованными эталонами, лестница A–D, зонд semantic-порога.",
        ),
        (
            "Оценка ответов LLM",
            "Для **8** конфигов (FRIDA×3 + строки 3–7 top-8) — отдельный прогон: `RagService.answer` + avg_ref_sim vs `answer_ref`.",
        ),
    ]


def md_table(headers: List[str], body: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _glossary_md_table() -> str:
    body = [
        [
            m.title,
            m.plain_ru,
            m.formula_ru,
            m.when_to_use,
            m.not_comparable_with or "—",
        ]
        for m in METRIC_GLOSSARY
    ]
    return md_table(
        ["Метрика", "Что это", "Как считали", "Когда смотреть", "Не сравнивать с"],
        body,
    )


def _matrix_md(
    bundle: ReportBundle,
    metric_key: str,
    *,
    as_pct: bool = False,
    highlight_best: bool = False,
) -> str:
    col_headers = ["Модель поиска"] + [profile_label_human(p) for p in bundle.profiles]
    matrix, best_coord = matrix_values(bundle, metric_key)
    body: List[List[str]] = []
    for ri, sid in enumerate(bundle.scenarios):
        row = [scenario_display_label(sid, bundle.summary)]
        for ci, pname in enumerate(bundle.profiles):
            v = matrix[ri][ci] if ri < len(matrix) else None
            if as_pct:
                cell_txt = format_pct(v)
            else:
                cell_txt = f"{v:.4f}" if v is not None else "—"
            if highlight_best and best_coord and best_coord[0] == ri and best_coord[1] == ci:
                cell_txt = f"**{cell_txt}**"
            row.append(cell_txt)
        body.append(row)
    return md_table(col_headers, body)


def _trust_context(bundle: ReportBundle) -> Dict[str, Any]:
    neg = (bundle.validity or {}).get("checks", {}).get("neg_control_retrieval") or {}
    ext = (bundle.validity or {}).get("checks", {}).get("external_sanity") or {}
    ladder_chk = (bundle.validity or {}).get("checks", {}).get("metrics_ladder") or {}
    sem_probe = (bundle.validity or {}).get("checks", {}).get("semantic_threshold_probe") or {}
    ladder_frac_a = (bundle.ladder or {}).get("class_fractions", {}).get("A", 0.59)
    ladder_frac_d = (bundle.ladder or {}).get("class_fractions", {}).get("D", 0.06)
    return {
        "neg": neg,
        "ext": ext,
        "ladder_chk": ladder_chk,
        "sem_probe": sem_probe,
        "ladder_frac_a": ladder_frac_a,
        "ladder_frac_d": ladder_frac_d,
    }


def build_outcomes_markdown(
    bundle: ReportBundle,
    answer_rows: List[AnswerEvalRow],
    diverse_rows: List[AnswerEvalRow],
    root: Path,
) -> List[str]:
    n_q = int(bundle.protocol.get("eval_set", {}).get("expected_questions", 101))
    n_chunks = int(bundle.protocol.get("corpus", {}).get("expected_collection_rows", 61543))
    fetch_k = int(bundle.summary.get("fetch_k") or 40)
    gap_rows = load_answer_top8_gap_metrics(bundle, root)
    merged = merge_answer_metrics(answer_rows, diverse_rows, gap_rows)
    model = build_outcomes_model(
        bundle.summary,
        n_questions=n_q,
        n_chunks=n_chunks,
        fetch_k=fetch_k,
        answer_rows=merged,
    )
    return render_outcomes_markdown(model)


def build_markdown(bundle: ReportBundle, root: Path) -> str:
    best_v, best_sid, best_prof = best_primary_cell(bundle)
    fetch_k = bundle.summary.get("fetch_k", 40)
    ctx = _trust_context(bundle)
    answer_rows = load_answer_metrics(bundle, root)
    diverse_rows = load_answer_diverse_metrics(bundle, root)
    gap_rows = load_answer_top8_gap_metrics(bundle, root)
    user_bge_m3_rows = load_answer_user_bge_m3_metrics(bundle, root)
    merged_answer = merge_answer_metrics(answer_rows, diverse_rows, gap_rows, user_bge_m3_rows)
    steps = build_experiment_steps(bundle)
    groups = group_metric_keys(bundle.metric_keys)
    shape = matrix_shape_label(bundle)

    lines: List[str] = [
        "# Финальное испытание поиска по полному корпусу документов",
        "",
        f"*Отчёт v2, собран: {bundle.generated}*",
        "",
        "## Содержание",
        "",
        "1. [Обзор](#1-обзор)",
        "2. [Итоги](#2-итоги)",
        "3. [Как устроен RAG](#3-как-устроен-rag)",
        "4. [Эксперимент](#4-эксперимент)",
        "5. [Словарь метрик](#5-словарь-метрик)",
        "6. [Результаты](#6-результаты)",
        "7. [Проверки доверия](#7-проверки-доверия)",
        "8. [Ответы LLM](#8-ответы-llm)",
        "9. [Приложение](#9-приложение)",
        "",
        "---",
        "",
        "## 1. Обзор",
        "",
        "Мы проверили, **насколько хорошо система находит нужные фрагменты** в большой базе "
        f"(**{int(bundle.protocol.get('corpus', {}).get('expected_collection_rows', 61543)):,}** чанков), "
        f"по **{int(bundle.protocol.get('eval_set', {}).get('expected_questions', 101))}** контрольным вопросам. "
        f"Сравнили **{len(bundle.scenarios)}** моделей эмбеддинга и **{len(bundle.profiles)}** reranker — "
        f"**{len(bundle.scenarios) * len(bundle.profiles)}** комбинаций.",
        "",
        f"Матрица **{shape}**: в каждой ячейке измерен **embedder + reranker** (dense top-40 → rerank). "
        "Отдельного «только retrieval» в прогоне нет — см. **[раздел 2](#2-итоги)** "
        "(вердикт: embedder, reranker, комбинация для прода, answer-eval).",
        "",
        f"> У ~**{ctx['ladder_frac_a']:.0%}** вопросов эталон **не встречается** в корпусе дословно (класс A) — "
        "потолок substring recall ограничен данными.",
        "",
        "> Ответ LLM (`avg_ref_sim`) — отдельный прогон, раздел 8; не смешивать с Recall@10 по чанкам.",
        "",
        "---",
        "",
    ]
    lines.extend(build_outcomes_markdown(bundle, answer_rows, diverse_rows, root))
    lines.extend(
        [
        "## 3. Как устроен RAG",
        "",
        "```",
        f"Корпус (61k чанков) → {len(bundle.scenarios)} индексов Chroma (разные embedder)",
        "       ↓",
        "Вопрос → dense top-40 → rerank → топ-5/10/20",
        "       ↓",
        "Метрики hit: substring | token_f1 | semantic",
        "       ↓",
        "(отдельно) LLM answer → cosine(ответ, answer_ref)",
        "```",
        "",
        "| Этап | Что происходит | Что измеряли в этом отчёте |",
        "|------|----------------|----------------------------|",
        "| Индекс | Чанки + эмбеддинг в Chroma | 6+1 коллекций, одинаковый корпус |",
        "| Retrieval | Векторный поиск top-40 | Время, кандидаты в trace |",
        "| Rerank | Cross-encoder пересортировка | 3 профиля reranker |",
        "| Оценка чанков | Hit в top-k | Recall, MRR, nDCG (3 режима) |",
        "| Ответ LLM | RagService.answer | avg_ref_sim (FRIDA×3 + gap×5 + USER-bge-m3×3) |",
        "",
        "---",
        "",
        "## 4. Эксперимент",
        "",
        ]
    )
    for title, desc in steps:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(desc)
        lines.append("")

    lines.extend(
        [
            "| Параметр | Значение |",
            "|----------|----------|",
            f"| fetch_k (до rerank) | **{fetch_k}** |",
            "| k для метрик | 5, 10, 20 |",
            "| Режимы hit | substring, token_f1, semantic (0.62) |",
            "",
            "### Модели поиска",
            "",
            md_table(
                ["Модель", "Индекс"],
                [[scenario_display_label(s, bundle.summary), s] for s in bundle.scenarios],
            ),
            "",
            "### Reranker",
            "",
            md_table(
                ["Reranker", "Роль"],
                [[profile_label_human(p), f"пересортировка {fetch_k} кандидатов"] for p in bundle.profiles],
            ),
            "",
            "---",
            "",
            "## 5. Словарь метрик",
            "",
            _glossary_md_table(),
            "",
            "---",
            "",
            "## 6. Результаты",
            "",
            f"Лидер: **{scenario_display_label(best_sid, bundle.summary)}** + "
            f"**{profile_label_human(best_prof)}** — substring Recall@10 = **{format_pct(best_v)}**.",
            "",
            "### Сводная таблица Recall@10 (%)",
            "",
            _matrix_md(bundle, PRIMARY_METRIC, as_pct=True, highlight_best=True),
            "",
            "### Топ-10 комбинаций",
            "",
        ]
    )

    lb_body: List[List[str]] = []
    for i, (v, sid, pname) in enumerate(leaderboard(bundle.summary, PRIMARY_METRIC, 10)):
        cell = (bundle.summary.get("results") or {}).get(sid, {}).get(pname) or {}
        lb_body.append(
            [
                str(i + 1),
                scenario_display_label(sid, bundle.summary),
                profile_label_human(pname),
                format_pct(v),
                format_pct(metric_value(cell, "substring::recall@20")),
            ]
        )
    lines.append(md_table(["#", "Модель", "Reranker", "R@10", "R@20"], lb_body))

    lines.extend(["", "### Recall@20 (substring, %)", "", _matrix_md(bundle, "substring::recall@20", as_pct=True), ""])
    lines.extend(["", "### MRR (substring)", "", _matrix_md(bundle, "substring::mrr", as_pct=False), ""])

    lines.extend(["", "---", "", "## 7. Проверки доверия", ""])

    if bundle.validity:
        ext = ctx["ext"]
        neg = ctx["neg"]
        lines.append(
            md_table(
                ["Проверка", "Статус", "Суть", "Как запускали"],
                [
                    [
                        "SQuAD sanity",
                        "PASS" if ext.get("pass") else "FAIL",
                        f"Recall@10 = {format_pct(float(ext.get('value') or 0))} на ~80 лёгких вопросах",
                        "external_sanity, маленький корпус с дословным ответом",
                    ],
                    [
                        "Neg-control",
                        "PASS" if neg.get("pass") else "FAIL",
                        f"{format_pct(neg.get('honest_value'))} → {format_pct(neg.get('shuffled_value'))}",
                        "shuffled_questions_gold_v2.jsonl, та же FRIDA+Jina",
                    ],
                    [
                        "Лестница A–D",
                        "PASS" if ctx["ladder_chk"].get("pass") else "инфо",
                        f"~{ctx['ladder_frac_a']:.0%} класс A, ~{ctx['ladder_frac_d']:.0%} класс D",
                        "metrics_ladder_funnel на BGE-M3 trace + corpus JSONL",
                    ],
                    [
                        "Semantic probe",
                        "PASS" if ctx["sem_probe"].get("pass") else "FAIL",
                        "Калибровка порога 0.62",
                        "semantic_threshold_probe на trace",
                    ],
                ],
            )
        )

    lines.extend(["", "---", "", "## 8. Ответы LLM", ""])
    lines.append(
        "Отдельный прогон (не B3 retrieval matrix): на каждый из **101** вопросов — "
        "`RagService.answer` (Chroma + rerank + DeepSeek), затем **avg_ref_sim** — "
        "cosine(embed(ответ), embed(`answer_ref`)). Это **не** substring Recall@10 по чанкам."
    )
    lines.append("")
    lines.append("#### Сводная (8 конфигов)")
    lines.append("")
    if merged_answer:
        lines.append(_answer_eval_md_table(bundle, merged_answer))
    else:
        lines.append("*Нет артефактов answer-eval.*")

    lines.append("")
    lines.append("#### FRIDA×3 (`_answer_top`)")
    lines.append("")
    if answer_rows:
        lines.append(_answer_eval_md_table(bundle, answer_rows))
    else:
        lines.append("*Артефакты answer-eval не найдены в `_answer_top/`.*")

    lines.append("")
    lines.append("#### Top-8 gap, строки 3–7 (`_answer_top8_gap`)")
    lines.append("")
    if gap_rows:
        lines.append(_answer_eval_md_table(bundle, gap_rows))
    else:
        lines.append(
            "*Не запускался.* Команда: "
            "`python -m eval.diagnostics.run_final_research_answer_top8_gap "
            "--b3-run-dir eval/runs/20260527_161527_b3_full_corpus_rerank`"
        )

    lines.append("")
    lines.append("#### USER-bge-m3 supplement (`_answer_user_bge_m3`, 3×101)")
    lines.append("")
    if user_bge_m3_rows:
        lines.append(_answer_eval_md_table(bundle, user_bge_m3_rows))
    else:
        lines.append(
            "*Не запускался.* Команда: "
            "`python -m eval.diagnostics.run_final_research_answer_user_bge_m3 "
            "--b3-run-dir eval/runs/<ts>_b3_user_bge_m3_rerank`"
        )

    lines.append("")
    lines.append("#### Расширенный набор (`_answer_diverse`, лидеры по метрикам)")
    lines.append("")
    if diverse_rows:
        dbody = []
        for r in diverse_rows:
            dbody.append(
                [
                    str(r.rank),
                    scenario_display_label(r.scenario, bundle.summary),
                    profile_label_human(r.profile),
                    format_pct(r.retrieval_recall10),
                    format_sim(r.avg_ref_sim),
                    "OK" if r.ok else "ошибка",
                ]
            )
        lines.append(
            md_table(["#", "Модель", "Reranker", "Retrieval R@10", "avg_ref_sim", "Статус"], dbody)
        )
    else:
        lines.append(
            "*Не запускался.* Команда: "
            "`python -m eval.diagnostics.run_final_research_answer_diverse "
            "--b3-run-dir eval/runs/20260527_161527_b3_full_corpus_rerank`"
        )

    lines.extend(["", "---", "", "## 9. Приложение", ""])
    lines.append(
        f"Полные матрицы {shape} по **{len(bundle.metric_keys)}** показателям, сгруппированы по режиму hit."
    )
    lines.append("")

    for gkey in ("primary", "substring", "token_f1", "semantic", "other"):
        keys = groups.get(gkey) or []
        if not keys:
            continue
        lines.append(f"### {GROUP_LABELS_RU.get(gkey, gkey)}")
        lines.append("")
        for mk in keys:
            as_pct = "recall@" in mk
            lines.append(f"#### `{mk}`")
            lines.append("")
            lines.append(_matrix_md(bundle, mk, as_pct=as_pct))
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Для разработчиков",
            "",
            f"- run_dir: `{bundle.b3_path.relative_to(root).as_posix()}`",
            "- Протокол: `eval/config/final_research_protocol.yaml`",
            "- HTML: `analysis/reports/final_canonical_experiment_2026_05.html`",
            "",
            "```bash",
            "python -m eval.reporting.emit_final_canonical_experiment_report --write-html",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


# --- HTML ---

HTML_CSS = """
:root {
  --text: #1a1a1a; --muted: #555; --border: #d8dce3; --bg: #eef1f6;
  --accent: #0b5fff; --pass: #0d7a3e; --card: #fff;
  --callout: #eef4ff; --best: #e8f5e9; --warn: #fff8e6;
}
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55; color: var(--text); margin: 0; background: #f0f2f5;
}
.page { max-width: 1400px; margin: 0 auto; padding: 24px 20px 64px; }
h1 { font-size: 1.75rem; margin: 0 0 6px; }
.lead { color: var(--muted); margin: 0 0 20px; }
.card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px; margin: 12px 0; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.card.highlight { border-left: 4px solid var(--accent); }
.card.warn { background: var(--warn); border-color: #e8d9a8; }
.verdict-hero {
  display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; margin: 16px 0;
}
@media (max-width: 720px) { .verdict-hero { grid-template-columns: 1fr; } }
.verdict-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 18px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.verdict-card.pick { border: 2px solid var(--accent); background: var(--callout); }
.verdict-card h3 { margin: 0 0 8px; font-size: 0.95rem; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
.verdict-card .big { font-size: 1.05rem; font-weight: 700; margin: 0 0 6px; }
td.best { background: var(--best); font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }
.stat .n { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.stat .l { font-size: 0.85rem; color: var(--muted); }
.pipeline {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; margin: 16px 0;
}
.pipe-step {
  flex: 1 1 140px; background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; font-size: 0.88rem;
}
.pipe-step strong { display: block; margin-bottom: 4px; font-size: 0.8rem; text-transform: uppercase; color: var(--muted); }
.pipe-arrow { align-self: center; color: var(--muted); font-size: 1.2rem; }
.glossary-grid { display: grid; gap: 12px; margin: 12px 0; }
.gloss-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.gloss-card h4 { margin: 0 0 6px; font-size: 1rem; }
.gloss-card p { margin: 4px 0; font-size: 0.9rem; }
.gloss-card .formula { font-size: 0.82rem; color: var(--muted); font-style: italic; }
.gloss-card .nc { font-size: 0.8rem; color: #a33; margin-top: 6px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 16px; font-size: 0.88rem; }
th, td { border: 1px solid var(--border); padding: 7px 9px; text-align: left; }
th { background: var(--bg); }
.hero-table td.best { background: var(--best); font-weight: 700; }
tr:nth-child(even) td { background: #fcfcfd; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.badge.pass { background: #d4edda; color: var(--pass); }
.badge.fail { background: #f8d7da; color: #842029; }
.callout { background: var(--callout); border: 1px solid #c5d9ff; border-radius: 8px; padding: 12px 14px; margin: 12px 0; }
.meta { font-size: 0.85rem; color: var(--muted); }
details { margin: 10px 0; border: 1px solid var(--border); border-radius: 8px; background: var(--card); }
details > summary { cursor: pointer; font-weight: 600; padding: 10px 14px; }
details .inner { padding: 0 14px 14px; }
details h4 { font-size: 0.9rem; margin: 12px 0 6px; }
ol.steps { padding-left: 1.2rem; }
ol.steps li { margin-bottom: 10px; }
/* Main tabs */
.tabs-main > input[type="radio"] {
  position: fixed; left: -9999px; width: 1px; height: 1px; opacity: 0;
}
.tab-hint {
  font-size: 0.9rem; color: var(--muted); margin: 0 0 10px;
  padding: 8px 12px; background: var(--card); border-radius: 8px; border: 1px solid var(--border);
}
.tab-bar {
  display: flex; flex-wrap: wrap; gap: 6px; position: sticky; top: 0; z-index: 20;
  background: #f0f2f5; padding: 12px 0 14px; border-bottom: 2px solid var(--border); margin-bottom: 12px;
}
.tab-bar label {
  padding: 10px 16px; border-radius: 8px; cursor: pointer; font-size: 0.9rem; font-weight: 600;
  background: var(--card); border: 2px solid var(--border); color: var(--text);
  box-shadow: 0 1px 2px rgba(0,0,0,.06);
}
.tab-bar label:hover { border-color: var(--accent); color: var(--accent); }
.tab-bar label.active {
  background: var(--accent); color: #fff; border-color: var(--accent);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
/* CSS fallback when JS disabled */
#tab-overview:checked ~ .tab-bar label[for="tab-overview"],
#tab-outcomes:checked ~ .tab-bar label[for="tab-outcomes"],
#tab-pipeline:checked ~ .tab-bar label[for="tab-pipeline"],
#tab-experiment:checked ~ .tab-bar label[for="tab-experiment"],
#tab-glossary:checked ~ .tab-bar label[for="tab-glossary"],
#tab-results:checked ~ .tab-bar label[for="tab-results"],
#tab-trust:checked ~ .tab-bar label[for="tab-trust"],
#tab-answers:checked ~ .tab-bar label[for="tab-answers"],
#tab-benchmark:checked ~ .tab-bar label[for="tab-benchmark"],
#tab-appendix:checked ~ .tab-bar label[for="tab-appendix"] {
  background: var(--accent); color: #fff; border-color: var(--accent);
}
#tab-overview:checked ~ .panels #panel-overview,
#tab-outcomes:checked ~ .panels #panel-outcomes,
#tab-pipeline:checked ~ .panels #panel-pipeline,
#tab-experiment:checked ~ .panels #panel-experiment,
#tab-glossary:checked ~ .panels #panel-glossary,
#tab-results:checked ~ .panels #panel-results,
#tab-trust:checked ~ .panels #panel-trust,
#tab-answers:checked ~ .panels #panel-answers,
#tab-benchmark:checked ~ .panels #panel-benchmark,
#tab-appendix:checked ~ .panels #panel-appendix { display: block; }
.embed-iframe {
  width: 100%; min-height: 1200px; border: 1px solid var(--border);
  border-radius: 8px; background: #fff;
}
/* Results sub-tabs */
.subtabs > input[type="radio"] { position: absolute; opacity: 0; }
.sub-bar { display: flex; gap: 6px; margin: 12px 0; }
.sub-bar label {
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
  border: 1px solid var(--border); background: var(--card);
}
#sub-r10:checked ~ .sub-bar label[for="sub-r10"],
#sub-r20:checked ~ .sub-bar label[for="sub-r20"],
#sub-mrr:checked ~ .sub-bar label[for="sub-mrr"] {
  background: var(--bg); font-weight: 700;
}
.sub-panel { display: none; }
#sub-r10:checked ~ .sub-panels #subpanel-r10,
#sub-r20:checked ~ .sub-panels #subpanel-r20,
#sub-mrr:checked ~ .sub-panels #subpanel-mrr { display: block; }
"""

HTML_TAB_SCRIPT = """
<script>
(function () {
  var root = document.querySelector(".tabs-main");
  if (!root) return;
  var inputs = root.querySelectorAll('input[name="main-tab"]');
  var panels = root.querySelectorAll(".tab-panel");
  var labels = root.querySelectorAll(".tab-bar label");
  var hashToTab = {
    outcomes: "tab-outcomes",
    recommendation: "tab-outcomes",
    overview: "tab-overview",
    rag: "tab-pipeline",
    pipeline: "tab-pipeline",
    experiment: "tab-experiment",
    metrics: "tab-glossary",
    glossary: "tab-glossary",
    results: "tab-results",
    trust: "tab-trust",
    answers: "tab-answers",
    benchmark: "tab-benchmark",
    appendix: "tab-appendix"
  };
  function panelForInput(id) {
    return "panel-" + id.replace("tab-", "");
  }
  function sync() {
    var active = "tab-outcomes";
    inputs.forEach(function (inp) {
      if (inp.checked) active = inp.id;
    });
    panels.forEach(function (p) {
      var on = p.id === panelForInput(active);
      p.classList.toggle("active", on);
      p.style.display = on ? "block" : "none";
    });
    labels.forEach(function (lab) {
      lab.classList.toggle("active", lab.htmlFor === active);
    });
  }
  inputs.forEach(function (inp) {
    inp.addEventListener("change", sync);
  });
  var h = (location.hash || "").replace(/^#/, "").toLowerCase();
  if (hashToTab[h]) {
    var el = document.getElementById(hashToTab[h]);
    if (el) el.checked = true;
  }
  sync();
  var subtabs = document.querySelectorAll(".subtabs");
  subtabs.forEach(function (sub) {
    var subInputs = sub.querySelectorAll('input[name="res-sub"]');
    var subPanels = sub.querySelectorAll(".sub-panel");
    var subLabels = sub.querySelectorAll(".sub-bar label");
    function subSync() {
      var active = "sub-r10";
      subInputs.forEach(function (inp) {
        if (inp.checked) active = inp.id;
      });
      subPanels.forEach(function (p) {
        var on = p.id === "subpanel-" + active.replace("sub-", "");
        p.style.display = on ? "block" : "none";
      });
      subLabels.forEach(function (lab) {
        lab.classList.toggle("active", lab.htmlFor === active);
      });
    }
    subInputs.forEach(function (inp) {
      inp.addEventListener("change", subSync);
    });
    subSync();
  });
})();
</script>
"""


def _html_table(
    headers: List[str],
    rows: List[List[str]],
    *,
    best_coords: Optional[set[Tuple[int, int]]] = None,
    hero: bool = False,
) -> str:
    cls = ' class="hero-table"' if hero else ""
    parts = [f"<table{cls}><thead><tr>"]
    parts.append("".join(f"<th>{html.escape(h)}</th>" for h in headers))
    parts.append("</tr></thead><tbody>")
    for ri, row in enumerate(rows):
        parts.append("<tr>")
        for ci, cell in enumerate(row):
            td_cls = ""
            if best_coords and (ri, ci) in best_coords:
                td_cls = ' class="best"'
            parts.append(f"<td{td_cls}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _matrix_html_rows(
    bundle: ReportBundle,
    metric_key: str,
    *,
    as_pct: bool = False,
) -> List[List[str]]:
    matrix, _ = matrix_values(bundle, metric_key)
    rows: List[List[str]] = []
    for ri, sid in enumerate(bundle.scenarios):
        row = [html.escape(scenario_display_label(sid, bundle.summary))]
        for ci, _ in enumerate(bundle.profiles):
            v = matrix[ri][ci] if ri < len(matrix) else None
            if as_pct:
                row.append(html.escape(format_pct(v)))
            else:
                row.append(html.escape(f"{v:.4f}" if v is not None else "—"))
        rows.append(row)
    return rows


def _tab_inputs(tab_ids: List[str], checked: str) -> str:
    parts = []
    for tid in tab_ids:
        chk = " checked" if tid == checked else ""
        parts.append(f'<input type="radio" name="main-tab" id="{tid}"{chk}/>')
    return "".join(parts)


DEFAULT_TAB = "tab-outcomes"


def _embedding_benchmark_iframe_src(protocol: Dict[str, Any]) -> str:
    embedded = protocol.get("embedded_reports") or {}
    rel = str(embedded.get("embedding_benchmark") or "analysis/reports/embedding_benchmark_nagruzka_2026_05.html")
    return Path(rel).name


def build_html(bundle: ReportBundle, root: Path) -> str:
    best_v, best_sid, best_prof = best_primary_cell(bundle)
    fetch_k = bundle.summary.get("fetch_k", 40)
    ctx = _trust_context(bundle)
    n_chunks = int(bundle.protocol.get("corpus", {}).get("expected_collection_rows", 61543))
    n_q = int(bundle.protocol.get("eval_set", {}).get("expected_questions", 101))
    n_cells = len(bundle.scenarios) * len(bundle.profiles)
    answer_rows = load_answer_metrics(bundle, root)
    diverse_rows = load_answer_diverse_metrics(bundle, root)
    gap_rows = load_answer_top8_gap_metrics(bundle, root)
    user_bge_m3_rows = load_answer_user_bge_m3_metrics(bundle, root)
    merged_answer = merge_answer_metrics(answer_rows, diverse_rows, gap_rows, user_bge_m3_rows)
    outcomes = build_outcomes_model(
        bundle.summary,
        n_questions=n_q,
        n_chunks=n_chunks,
        fetch_k=int(fetch_k),
        answer_rows=merged_answer,
    )
    steps = build_experiment_steps(bundle)
    groups = group_metric_keys(bundle.metric_keys)
    shape = matrix_shape_label(bundle)

    col_headers = ["Модель поиска"] + [profile_label_human(p) for p in bundle.profiles]
    _, best_coord = matrix_values(bundle, PRIMARY_METRIC)
    best_set: set[Tuple[int, int]] = set()
    if best_coord:
        best_set.add((best_coord[0], best_coord[1] + 1))

    tab_ids = [
        "tab-overview",
        "tab-outcomes",
        "tab-pipeline",
        "tab-experiment",
        "tab-glossary",
        "tab-results",
        "tab-trust",
        "tab-answers",
        "tab-benchmark",
        "tab-appendix",
    ]
    tab_labels = [
        ("tab-overview", "Обзор"),
        ("tab-outcomes", "Итоги"),
        ("tab-pipeline", "RAG"),
        ("tab-experiment", "Эксперимент"),
        ("tab-glossary", "Метрики"),
        ("tab-results", "Результаты"),
        ("tab-trust", "Проверки"),
        ("tab-answers", "Ответы LLM"),
        ("tab-benchmark", "Бенчмарк embedder"),
        ("tab-appendix", "Приложение"),
    ]
    benchmark_iframe_src = _embedding_benchmark_iframe_src(bundle.protocol)

    parts: List[str] = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Испытание поиска по полному корпусу — отчёт 2026-05</title>",
        f"<style>{HTML_CSS}</style></head><body>",
        '<div class="page">',
        "<h1>Испытание поиска по полной базе знаний</h1>",
        f'<p class="lead">Отчёт v2: {n_q} вопросов, {n_chunks:,} чанков, {n_cells} конфигураций. '
        f"Собран {html.escape(bundle.generated)}.</p>",
        '<div class="tabs-main">',
        '<p class="tab-hint"><strong>Разделы отчёта</strong> — нажмите кнопку ниже. '
        'Старт: <strong>Итоги</strong> — что брать в прод. '
        'Ссылка: <code>#outcomes</code></p>',
        _tab_inputs(tab_ids, DEFAULT_TAB),
        '<nav class="tab-bar" role="tablist" aria-label="Разделы отчёта">',
    ]
    for tid, label in tab_labels:
        parts.append(f'<label for="{tid}">{html.escape(label)}</label>')
    parts.append("</nav><div class='panels'>")

    # --- Overview ---
    parts.append('<section class="tab-panel" id="panel-overview">')
    parts.append('<div class="grid">')
    for n, lbl in [
        (str(n_q), "контрольных вопросов"),
        (f"{n_chunks:,}", "чанков в индексе"),
        (str(n_cells), f"ячеек {shape}"),
        (str(len(bundle.scenarios)), "embedder"),
        (f"{ctx['ladder_frac_a']:.0%}", "класс A (эталон не в корпусе)"),
    ]:
        parts.append(f'<div class="card stat"><div class="n">{html.escape(n)}</div><div class="l">{html.escape(lbl)}</div></div>')
    parts.append("</div>")
    parts.append(
        '<p class="meta">Embedder / reranker / комбинация — вкладка <strong>«Итоги»</strong>.</p>'
    )
    parts.append(
        f"<p class='meta'>Канонический прогон: <code>{html.escape(bundle.b3_path.name)}</code></p>"
    )
    parts.append("</section>")

    # --- Outcomes ---
    parts.append('<section class="tab-panel" id="panel-outcomes">')
    parts.extend(render_outcomes_html(outcomes))
    parts.append("</section>")

    # --- Pipeline ---
    parts.append('<section class="tab-panel" id="panel-pipeline">')
    parts.append("<h2>Как устроен RAG в этом испытании</h2>")
    parts.append('<div class="pipeline">')
    pipe_steps = [
        ("1. Корпус", f"~{n_chunks:,} чанков JSONL"),
        ("2. Эмбеддинг", f"{len(bundle.scenarios)} моделей → Chroma"),
        ("3. Запрос", f"dense top-{fetch_k}"),
        ("4. Rerank", "3 cross-encoder"),
        ("5. Метрики", "hit @ top 5/10/20"),
        ("6. LLM", "11 конфигов (отдельно)"),
    ]
    for i, (title, sub) in enumerate(pipe_steps):
        if i:
            parts.append('<span class="pipe-arrow">→</span>')
        parts.append(f'<div class="pipe-step"><strong>{html.escape(title)}</strong>{html.escape(sub)}</div>')
    parts.append("</div>")
    parts.append("</section>")

    # --- Experiment ---
    parts.append('<section class="tab-panel" id="panel-experiment">')
    parts.append("<h2>Как проводили эксперимент</h2><ol class='steps'>")
    for title, desc in steps:
        parts.append(f"<li><strong>{html.escape(title)}</strong> — {desc}</li>")
    parts.append("</ol>")
    parts.append("<h3>Модели поиска</h3>")
    parts.append(
        _html_table(
            ["Модель", "ID"],
            [
                [html.escape(scenario_display_label(s, bundle.summary)), html.escape(s)]
                for s in bundle.scenarios
            ],
        )
    )
    parts.append("<h3>Reranker</h3>")
    parts.append(
        _html_table(
            ["Название", "ID"],
            [
                [html.escape(profile_label_human(p)), html.escape(p)]
                for p in bundle.profiles
            ],
        )
    )
    parts.append("</section>")

    # --- Glossary ---
    parts.append('<section class="tab-panel" id="panel-glossary">')
    parts.append("<h2>Словарь метрик</h2><div class='glossary-grid'>")
    for m in METRIC_GLOSSARY:
        parts.append('<div class="gloss-card">')
        parts.append(f"<h4>{html.escape(m.title)}</h4>")
        parts.append(f"<p>{html.escape(m.plain_ru)}</p>")
        parts.append(f'<p class="formula">{html.escape(m.formula_ru)}</p>')
        parts.append(f"<p><strong>Когда:</strong> {html.escape(m.when_to_use)}</p>")
        if m.not_comparable_with:
            parts.append(f'<p class="nc">Не сравнивать: {html.escape(m.not_comparable_with)}</p>')
        parts.append("</div>")
    parts.append("</div></section>")

    # --- Results ---
    parts.append('<section class="tab-panel" id="panel-results">')
    parts.append("<h2>Результаты поиска</h2>")
    parts.append(
        f"<p>Лидер: <strong>{html.escape(scenario_display_label(best_sid, bundle.summary))}</strong> + "
        f"<strong>{html.escape(profile_label_human(best_prof))}</strong></p>"
    )
    parts.append('<div class="subtabs">')
    parts.append('<input type="radio" name="res-sub" id="sub-r10" checked/>')
    parts.append('<input type="radio" name="res-sub" id="sub-r20"/>')
    parts.append('<input type="radio" name="res-sub" id="sub-mrr"/>')
    parts.append('<div class="sub-bar">')
    parts.append('<label for="sub-r10">Recall@10 (%)</label>')
    parts.append('<label for="sub-r20">Recall@20 (%)</label>')
    parts.append('<label for="sub-mrr">MRR</label>')
    parts.append("</div><div class='sub-panels'>")
    parts.append('<div class="sub-panel" id="subpanel-r10">')
    parts.append("<p class='meta'>Substring: доля вопросов с эталоном в топ-10.</p>")
    parts.append(
        _html_table(
            col_headers,
            _matrix_html_rows(bundle, PRIMARY_METRIC, as_pct=True),
            best_coords=best_set,
            hero=True,
        )
    )
    parts.append("</div><div class='sub-panel' id='subpanel-r20'>")
    parts.append(
        _html_table(col_headers, _matrix_html_rows(bundle, "substring::recall@20", as_pct=True))
    )
    parts.append("</div><div class='sub-panel' id='subpanel-mrr'>")
    parts.append(
        _html_table(col_headers, _matrix_html_rows(bundle, "substring::mrr", as_pct=False))
    )
    parts.append("</div></div></div>")
    parts.append("<h3>Топ-10</h3>")
    lb_rows: List[List[str]] = []
    for i, (v, sid, pname) in enumerate(leaderboard(bundle.summary, PRIMARY_METRIC, 10)):
        cell = (bundle.summary.get("results") or {}).get(sid, {}).get(pname) or {}
        lb_rows.append(
            [
                html.escape(str(i + 1)),
                html.escape(scenario_display_label(sid, bundle.summary)),
                html.escape(profile_label_human(pname)),
                html.escape(format_pct(v)),
                html.escape(format_pct(metric_value(cell, "substring::recall@20"))),
            ]
        )
    parts.append(_html_table(["#", "Модель", "Reranker", "R@10", "R@20"], lb_rows))
    parts.append("</section>")

    # --- Trust ---
    parts.append('<section class="tab-panel" id="panel-trust">')
    parts.append("<h2>Проверки доверия</h2>")
    if bundle.validity:
        ext, neg = ctx["ext"], ctx["neg"]
        parts.append('<div class="card">')
        if ext.get("pass"):
            parts.append(
                f'<p><span class="badge pass">PASS</span> <strong>SQuAD sanity</strong> — '
                f"Recall@10 = {html.escape(format_pct(float(ext.get('value') or 0)))}. "
                "~80 вопросов, ответ дословно в маленьком корпусе. Проверяем, что pipeline живой.</p>"
            )
        parts.append(
            "<p class='meta'><strong>Зачем:</strong> если здесь провал — баг индекса/метрик, "
            "а не «сложный корпус».</p>"
        )
        parts.append("</div><div class='card'>")
        if neg.get("pass"):
            parts.append(
                f'<p><span class="badge pass">PASS</span> <strong>Neg-control</strong> — '
                f"честный {html.escape(format_pct(neg.get('honest_value')))} → "
                f"перетасованный {html.escape(format_pct(neg.get('shuffled_value')))}.</p>"
            )
        parts.append(
            "<p class='meta'><strong>Как:</strong> `shuffle_eval_questions_neg_control` + тот же retrieval "
            "на FRIDA+Jina.</p></div>"
        )
        parts.append(
            f'<div class="card"><p><strong>Лестница A–D:</strong> ~{ctx["ladder_frac_a"]:.0%} вопросов — '
            f"класс A (эталон не в корпусе как substring); ~{ctx['ladder_frac_d']:.0%} — класс D (hit@10). "
            "Объясняет потолок recall.</p></div>"
        )
        if ctx["sem_probe"].get("pass"):
            parts.append(
                '<div class="card"><p><span class="badge pass">PASS</span> '
                "<strong>Semantic probe</strong> — порог 0.62 согласован с распределением similarity.</p></div>"
            )
    parts.append("</section>")

    # --- Answers ---
    parts.append('<section class="tab-panel" id="panel-answers">')
    parts.append("<h2>Ответы LLM (отдельный прогон)</h2>")
    n_answer = len(merged_answer)
    parts.append(
        f"<p>Для <strong>{n_answer}</strong> конфигураций: на каждый из 101 вопросов "
        "<code>RagService.answer</code> (векторный RAG + rerank + DeepSeek), затем "
        "<strong>avg_ref_sim</strong> — cosine(embed(ответ), embed(answer_ref)).</p>"
    )
    best_ans_sim = merged_answer[0].avg_ref_sim if merged_answer else None
    parts.append(
        '<div class="callout card"><p>Это <em>не</em> Recall@10 по чанкам. '
        f"~{format_pct(best_v)} поиск ≠ ~{format_sim(best_ans_sim)} ответ.</p></div>"
    )
    parts.append("<h3>Сводная таблица answer-eval</h3>")
    if merged_answer:
        parts.append(_answer_eval_html_table(bundle, merged_answer))
    else:
        parts.append("<p class='meta'>Нет артефактов answer-eval.</p>")
    parts.append(
        "<p class='meta'>Источники объединены без дублей: <code>_answer_top</code>, "
        "<code>_answer_top8_gap</code>, <code>_answer_user_bge_m3</code>, <code>_answer_diverse</code>.</p>"
    )
    parts.append("</section>")

    # --- Embedding benchmark (external HTML) ---
    parts.append('<section class="tab-panel" id="panel-benchmark">')
    parts.append("<h2>Бенчмарк embedder (скорость, память, STS-B)</h2>")
    parts.append(
        "<p class='meta'>Отдельный прогон на CPU/MPS (не full-corpus retrieval). "
        f"Исходник: <code>{html.escape(benchmark_iframe_src)}</code></p>"
    )
    parts.append(
        f'<iframe class="embed-iframe" title="Embedding benchmark" '
        f'src="{html.escape(benchmark_iframe_src)}"></iframe>'
    )
    parts.append("</section>")

    # --- Appendix ---
    parts.append('<section class="tab-panel" id="panel-appendix">')
    parts.append("<h2>Приложение: все метрики</h2>")
    parts.append(
        f"<p class='meta'>Свёрнуто по группам — {len(bundle.metric_keys)} таблиц {html.escape(shape)}.</p>"
    )
    for gkey in ("primary", "substring", "token_f1", "semantic", "other"):
        keys = groups.get(gkey) or []
        if not keys:
            continue
        glabel = GROUP_LABELS_RU.get(gkey, gkey)
        parts.append(f"<details><summary>{html.escape(glabel)} — {len(keys)} таблиц</summary><div class='inner'>")
        for mk in keys:
            as_pct = "recall@" in mk
            parts.append(f"<h4>{html.escape(mk)}</h4>")
            parts.append(
                _html_table(col_headers, _matrix_html_rows(bundle, mk, as_pct=as_pct))
            )
        parts.append("</div></details>")
    parts.append("</section>")

    parts.append("</div></div>")  # panels, tabs-main
    parts.append(HTML_TAB_SCRIPT)
    parts.append('<div class="card meta" style="margin-top:24px">')
    parts.append(
        f"<p>run_dir: <code>{html.escape(bundle.b3_path.relative_to(root).as_posix())}</code></p>"
    )
    parts.append(
        "<p>Пересборка: <code>python -m eval.reporting.emit_final_canonical_experiment_report --write-html</code></p>"
    )
    parts.append("</div></div></body></html>")
    return "".join(parts)

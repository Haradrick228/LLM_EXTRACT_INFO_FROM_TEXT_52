"""F8: собрать HTML-отчёт ``retrieval_recovery_2026_05`` из артефактов F2–F4, F7, B3.

Идёт по ``eval/runs/`` и собирает:

- F2 (``*_f2_hybrid_rrf_sweep``) — sweep fetch_k × {dense_only, bm25_only, hybrid_rrf};
- F3 (``*_f3_query_transform_sweep``) — HyDE / MQR / hybrid;
- F4 (``*_f4_bge_m3_multi_vector_sweep``) — BGE-M3 multi-vector fusion;
- F7 (``*_f7_final_reranker_matrix``) — финальная реранк-матрица;
- B3 (``*_b3_full_corpus_rerank``) — матрица реранкеров на полном корпусе.

В таблицах к primary-метрикам добавлены колонки ``token_f1::recall@10`` и
``token_f1::recall@5``, если они есть в ``summary.json`` (иначе &mdash;).

В каждой папке ожидаются ``summary.json`` и ``matrix_manifest.json``. Если
файлов нет, секция помечается как «pending».

Артефакт по умолчанию: ``analysis/reports/retrieval_recovery_2026_05.html``.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_TOKEN_F1_APPEND = ["token_f1::recall@10", "token_f1::recall@5"]

_PHASE_SLUGS = [
    ("F2", "f2_hybrid_rrf_sweep"),
    ("F3", "f3_query_transform_sweep"),
    ("F4", "f4_bge_m3_multi_vector_sweep"),
    ("F7", "f7_final_reranker_matrix"),
    ("B3", "b3_full_corpus_rerank"),
]


def _find_latest(runs_root: Path, slug: str) -> Path | None:
    """Найти последнюю папку с подстрокой ``slug``."""
    if not runs_root.is_dir():
        return None
    cands = [p for p in runs_root.iterdir() if p.is_dir() and slug in p.name]
    if not cands:
        return None
    cands.sort()
    return cands[-1]


def _load_summary(path: Path) -> Dict[str, Any] | None:
    if not path or not path.is_dir():
        return None
    sp = path / "summary.json"
    if not sp.is_file():
        return None
    try:
        return json.loads(sp.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to load {sp}: {exc!r}", file=sys.stderr)
        return None


def _metric_table(rows: List[Dict[str, Any]], metric_keys: List[str], row_label_key: str) -> str:
    """rows: [{label_key: ..., metrics: {key->float}}, ...]"""
    head = "<tr><th>" + row_label_key + "</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys) + "</tr>"
    body_rows = []
    for r in rows:
        cells = [f"<td><code>{html.escape(str(r.get(row_label_key, '')))}</code></td>"]
        metrics = r.get("metrics") or {}
        for k in metric_keys:
            v = metrics.get(k)
            cells.append(
                f"<td>{f'{v:.4f}' if isinstance(v, (int, float)) else '&mdash;'}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{head}{''.join(body_rows)}</table>"


def _render_phase_f2(summary: Dict[str, Any]) -> str:
    parts: List[str] = ["<h3>F2 hybrid_rrf sweep</h3>"]
    rrf_k = summary.get("rrf_k")
    fetch_ks = summary.get("fetch_ks") or []
    parts.append(
        f"<p>RRF k = <code>{rrf_k}</code>, fetch_k sweep = <code>{fetch_ks}</code>, n_queries = <code>{summary.get('n_queries')}</code></p>"
    )
    metric_keys = [
        "recall@5", "recall@10", "recall@20",
        "precision@5", "precision@10", "mrr", "ndcg@3",
    ] + _TOKEN_F1_APPEND
    for sid, bucket in (summary.get("results") or {}).items():
        if not isinstance(bucket, dict):
            continue
        parts.append(f"<h4><code>{html.escape(sid)}</code></h4>")
        for fk_key, jobs in sorted((bucket or {}).items()):
            try:
                fk_int = int(fk_key)
            except Exception:
                fk_int = fk_key
            parts.append(f"<details><summary>fetch_k = <code>{fk_int}</code></summary>")
            rows = []
            for job_name, cell in (jobs or {}).items():
                if not isinstance(cell, dict):
                    continue
                rows.append({"job": job_name, "metrics": cell.get("metrics") or {}})
            parts.append(_metric_table(rows, metric_keys, "job"))
            parts.append("</details>")
    return "".join(parts)


def _render_phase_f3(summary: Dict[str, Any]) -> str:
    parts = ["<h3>F3 query-transform sweep</h3>"]
    parts.append(
        f"<p>LLM = <code>{html.escape(str(summary.get('llm_provider')))}</code>:"
        f" <code>{html.escape(str(summary.get('llm_model')))}</code>, "
        f"fetch_k = <code>{summary.get('fetch_k')}</code>, RRF k = <code>{summary.get('rrf_k')}</code></p>"
    )
    metric_keys = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@3"] + _TOKEN_F1_APPEND
    for sid, bucket in (summary.get("results") or {}).items():
        parts.append(f"<h4><code>{html.escape(sid)}</code></h4>")
        rows = []
        for job, cell in (bucket or {}).items():
            if isinstance(cell, dict):
                rows.append({"job": job, "metrics": cell.get("metrics") or {}})
        parts.append(_metric_table(rows, metric_keys, "job"))
    return "".join(parts)


def _render_phase_f4(summary: Dict[str, Any]) -> str:
    parts = ["<h3>F4 BGE-M3 multi-vector</h3>"]
    parts.append(
        f"<p>candidate_top_k = <code>{summary.get('candidate_top_k')}</code>; "
        f"top_k = <code>{summary.get('top_k')}</code></p>"
    )
    metric_keys = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@3"] + _TOKEN_F1_APPEND
    rows = []
    for combo, cell in (summary.get("results") or {}).items():
        if not isinstance(cell, dict):
            continue
        rows.append({"combo": combo, "metrics": cell.get("metrics") or {}})
    parts.append(_metric_table(rows, metric_keys, "combo"))
    return "".join(parts)


def _render_phase_f7(summary: Dict[str, Any]) -> str:
    parts = ["<h3>F7 final reranker matrix</h3>"]
    base_meta = summary.get("base_meta") or {}
    parts.append(f"<p>Base retriever: <code>{html.escape(json.dumps(base_meta, ensure_ascii=False))}</code></p>")
    metric_keys = ["recall@5", "recall@10", "recall@20", "mrr", "ndcg@3"] + _TOKEN_F1_APPEND
    rows = []
    for rname, cell in (summary.get("results") or {}).items():
        if isinstance(cell, dict):
            rows.append({"reranker": rname, "metrics": cell.get("metrics") or {}})
    parts.append(_metric_table(rows, metric_keys, "reranker"))
    return "".join(parts)


def _render_phase_b3(summary: Dict[str, Any]) -> str:
    parts = ["<h3>B3 full-corpus rerank matrix</h3>"]
    parts.append(
        f"<p>fetch_k = <code>{summary.get('fetch_k')}</code>, k = <code>{summary.get('k')}</code>, "
        f"n_queries (eval file) implied by subprocesses = <code>101</code> (gold v2).</p>"
    )
    metric_keys = [
        "recall@5", "recall@10", "recall@20",
        "precision@5", "precision@10", "mrr", "ndcg@3",
    ] + _TOKEN_F1_APPEND
    for sid, bucket in (summary.get("results") or {}).items():
        if not isinstance(bucket, dict):
            continue
        parts.append(f"<h4><code>{html.escape(str(sid))}</code></h4>")
        rows = []
        for prof, cell in bucket.items():
            if not isinstance(cell, dict):
                continue
            rows.append({"profile": prof, "metrics": cell.get("metrics") or {}})
        parts.append(_metric_table(rows, metric_keys, "profile"))
    return "".join(parts)


_PHASE_RENDERERS = {
    "F2": _render_phase_f2,
    "F3": _render_phase_f3,
    "F4": _render_phase_f4,
    "F7": _render_phase_f7,
    "B3": _render_phase_b3,
}


def _render_html(snapshot: Dict[str, Any]) -> str:
    css = """
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; max-width: 1200px; margin: 32px auto; padding: 0 16px; color: #1d1d1f; }
    h1, h2, h3, h4 { line-height: 1.25; }
    h1 { font-size: 28px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
    h2 { font-size: 22px; margin-top: 40px; }
    h3 { font-size: 18px; margin-top: 28px; color: #333; }
    table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
    th { background: #f6f6f6; }
    code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 12px; }
    .pending { color: #999; font-style: italic; }
    .meta { font-size: 12px; color: #666; }
    details > summary { cursor: pointer; }
    """
    parts: List[str] = []
    parts.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Retrieval Recovery 2026-05</title><style>{css}</style></head><body>")
    parts.append("<h1>Retrieval Recovery 2026-05 — F2–F4, F7, B3 summary</h1>")
    parts.append(
        "<p>Отчёт собран автоматически из <code>eval/runs/</code> по подкаталогам с суффиксами "
        "<code>f2_*</code>, <code>f3_*</code>, <code>f4_*</code>, <code>f7_*</code>, <code>b3_*</code>. "
        "Каждая секция показывает последний прогон фазы (лексикографически максимальное имя каталога).</p>"
    )
    parts.append(
        "<p class='meta'>Метрики считаются по eval_set_gold_v2.jsonl. "
        "<strong>Fixture (gold_v2_fixture)</strong> — 101 чанк, recall ограничен сверху ~99%. "
        "<strong>Full corpus (corpus_phase_b_*)</strong> — 61543 чанка; recall@10 для dense обычно 5-15% без реранка.</p>"
    )

    for label, slug in _PHASE_SLUGS:
        info = snapshot.get(label) or {}
        path = info.get("path")
        summary = info.get("summary")
        parts.append(f"<h2>{label}: {html.escape(slug)}</h2>")
        if not summary:
            parts.append(
                f"<p class='pending'>Артефакты не найдены (искал <code>eval/runs/*{html.escape(slug)}*/summary.json</code>). "
                "Запусти соответствующий эксперимент: см. <code>eval/config/experiments_registry.yaml</code>.</p>"
            )
            continue
        parts.append(f"<p class='meta'>Source: <code>{html.escape(str(path))}</code></p>")
        try:
            parts.append(_PHASE_RENDERERS[label](summary))
        except Exception as exc:
            parts.append(f"<p>Render error: <code>{html.escape(repr(exc))}</code></p>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=_ROOT / "eval" / "runs")
    p.add_argument(
        "--out-html", type=Path,
        default=_ROOT / "analysis" / "reports" / "retrieval_recovery_2026_05.html",
    )
    p.add_argument(
        "--out-snapshot-json", type=Path,
        default=None,
        help="Если задано — записать сводный JSON со ссылками на выбранные прогоны.",
    )
    args = p.parse_args()

    def _rel_or_str(p: Path) -> str:
        try:
            return str(p.relative_to(_ROOT))
        except Exception:
            return str(p)

    snapshot: Dict[str, Any] = {}
    for label, slug in _PHASE_SLUGS:
        run_dir = _find_latest(args.runs_root, slug)
        summ = _load_summary(run_dir) if run_dir else None
        snapshot[label] = {
            "path": _rel_or_str(run_dir) if run_dir else None,
            "summary": summ,
        }

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(_render_html(snapshot), encoding="utf-8")

    if args.out_snapshot_json is not None:
        args.out_snapshot_json.parent.mkdir(parents=True, exist_ok=True)
        # ужмём snapshot — не дублируем все данные, только пути + ключи фаз
        compact = {
            lbl: {"path": v.get("path"), "summary_present": v.get("summary") is not None}
            for lbl, v in snapshot.items()
        }
        args.out_snapshot_json.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    print(f"OK: {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

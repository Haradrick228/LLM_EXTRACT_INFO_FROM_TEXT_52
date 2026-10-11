"""Smoke-тест ``eval.reporting.build_retrieval_recovery_report``.

Создаёт фейковые ``eval/runs/<ts>_f*_*`` каталоги с ``summary.json`` для F2/F3/F4/F7,
запускает CLI и проверяет, что HTML-отчёт собран и содержит таблицы.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _mk_summary(parent: Path, slug: str, content: dict) -> Path:
    d = parent / f"20260512_120000_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def test_build_report_assembles_html(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _mk_summary(runs_root, "f2_hybrid_rrf_sweep", {
        "n_queries": 10,
        "rrf_k": 60,
        "fetch_ks": [40, 100],
        "results": {
            "full_mpnet": {
                "40": {
                    "dense_only": {"metrics": {"recall@5": 0.1, "mrr": 0.2}, "ok": True},
                    "bm25_only":  {"metrics": {"recall@5": 0.08, "mrr": 0.18}, "ok": True},
                    "hybrid_rrf": {"metrics": {"recall@5": 0.13, "mrr": 0.25}, "ok": True},
                },
            }
        },
    })
    _mk_summary(runs_root, "f3_query_transform_sweep", {
        "n_queries": 10,
        "fetch_k": 100, "rrf_k": 60,
        "llm_provider": "ollama", "llm_model": "qwen2.5:7b",
        "results": {
            "full_mpnet": {
                "dense_only": {"metrics": {"recall@5": 0.10, "mrr": 0.20}, "ok": True},
                "hyde":       {"metrics": {"recall@5": 0.12, "mrr": 0.22}, "ok": True},
                "mqr":        {"metrics": {"recall@5": 0.13, "mrr": 0.25}, "ok": True},
                "hyde_mqr_dense_bm25_rrf": {"metrics": {"recall@5": 0.15, "mrr": 0.28}, "ok": True},
            }
        },
    })
    _mk_summary(runs_root, "f4_bge_m3_multi_vector_sweep", {
        "n_queries": 10,
        "candidate_top_k": 100, "top_k": 20,
        "results": {
            "balanced":    {"metrics": {"recall@5": 0.18, "mrr": 0.32}, "ok": True},
            "dense_only":  {"metrics": {"recall@5": 0.14, "mrr": 0.27}, "ok": True},
        },
    })
    _mk_summary(runs_root, "f7_final_reranker_matrix", {
        "n_queries": 10,
        "base_meta": {"kind": "hybrid_rrf", "base_id": "full_mpnet"},
        "results": {
            "__no_rerank__": {"metrics": {"recall@5": 0.13, "mrr": 0.25}, "ok": True},
            "BAAI/bge-reranker-v2-m3": {"metrics": {"recall@5": 0.45, "mrr": 0.58}, "ok": True},
        },
    })

    out_html = tmp_path / "out.html"
    cmd = [
        sys.executable, "-m", "eval.reporting.build_retrieval_recovery_report",
        "--runs-root", str(runs_root),
        "--out-html", str(out_html),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    text = out_html.read_text(encoding="utf-8")
    assert "Retrieval Recovery 2026-05" in text
    assert "F2 hybrid_rrf sweep" in text
    assert "F3 query-transform sweep" in text
    assert "F4 BGE-M3 multi-vector" in text
    assert "F7 final reranker matrix" in text
    # Численные значения из F7 должны быть в HTML
    assert "0.4500" in text  # recall@5 reranker
    assert "BAAI/bge-reranker-v2-m3" in text


def test_build_report_marks_missing_phases_as_pending(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # Никаких артефактов вообще
    out_html = tmp_path / "out.html"
    proc = subprocess.run(
        [
            sys.executable, "-m", "eval.reporting.build_retrieval_recovery_report",
            "--runs-root", str(runs_root),
            "--out-html", str(out_html),
        ],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    text = out_html.read_text(encoding="utf-8")
    assert "pending" in text.lower()
    for label in ("F2", "F3", "F4", "F7", "B3"):
        assert label in text

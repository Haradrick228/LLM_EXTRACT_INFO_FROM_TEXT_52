"""Smoke-тест на CLI ``eval.retrieval.run_hybrid_rrf_sweep``.

Не ходит в реальную Chroma — все компоненты заменяются стабами через
``--skip-bm25`` и monkeypatch. Цель — убедиться, что:

- запуск с реальным registry/eval-файлом проходит;
- summary.json/metrics_matrix.md создаются;
- evaluate_retriever получает корректные kwargs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def test_run_hybrid_rrf_sweep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Подменяем dense-build/chroma list и проверяем артефакты."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from core.rag.retrievers.retriever_base_types import (
        RetrievalRequest,
        RetrievalResult,
        RetrievedDoc,
    )
    from eval.retrieval import run_hybrid_rrf_sweep as sweep

    class _DummyDense:
        name = "dense_dummy"

        def search(self, request: RetrievalRequest) -> RetrievalResult:
            return RetrievalResult(
                docs=[
                    RetrievedDoc(
                        text="some text mentioning рефлексия explicitly",
                        score=0.9,
                        rank=1,
                        chunk_id="C1",
                    ),
                    RetrievedDoc(
                        text="totally unrelated content",
                        score=0.1,
                        rank=2,
                        chunk_id="C2",
                    ),
                ]
            )

    chroma_names = {
        "corpus_phase_b_mpnet",
        "corpus_phase_b_bge_m3",
        "corpus_phase_b_frida",
        "corpus_phase_b_e5_large_instruct",
        "corpus_phase_b_jina_embeddings_v3",
        "default",
    }

    class _DummyCollection:
        pass

    class _DummyChromaClient:
        def list_collections(self):
            return [type("C", (), {"name": n})() for n in chroma_names]

        def get_collection(self, name, embedding_function=None):
            return _DummyCollection()

    monkeypatch.setattr(sweep, "chroma_http_client", lambda **kw: _DummyChromaClient())
    monkeypatch.setattr(sweep, "build_chroma_embedding_function", lambda spec: None)
    monkeypatch.setattr(sweep, "_build_dense_retriever", lambda *a, **kw: _DummyDense())

    # tiny eval set
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "что такое рефлексия?", "references": ["рефлексия"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    # tmp run root
    monkeypatch.setattr(sweep, "_ROOT", tmp_path)

    argv = [
        "prog",
        "--dense-registry", str(REPO / "eval" / "config" / "embedding_bases_full_corpus.yaml"),
        "--scenarios", "full_mpnet",
        "--bm25-dir", str(tmp_path / "bm25_not_needed"),
        "--eval-file", str(eval_file),
        "--fetch-ks", "5",
        "--k", "1,3",
        "--rrf-k", "60",
        "--metric-modes", "substring",
        "--seed", "0",
        "--skip-bm25",
        "--max-rows", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = sweep.main()
    assert rc == 0

    runs_root = tmp_path / "eval" / "runs"
    assert runs_root.is_dir()
    run_dir = sorted(runs_root.iterdir())[-1]
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics_matrix.md").is_file()
    summ: Dict[str, Any] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cell = summ["results"]["full_mpnet"]["5"]["dense_only"]
    assert cell["ok"] is True
    # recall@1 should be 1.0 (single doc contains "рефлексия")
    assert cell["metrics"]["recall@1"] == pytest.approx(1.0)

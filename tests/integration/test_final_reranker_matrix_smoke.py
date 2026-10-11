"""Smoke-тест ``eval.retrieval.run_final_reranker_matrix`` без Chroma/HF-моделей.

Заменяем все внешние зависимости:

- ``_build_base_retriever`` — стаб, возвращающий фейк ретривер;
- ``build_reranker`` — стаб ScoreInjectorReranker (cross-encoder mock);
- ``chroma_http_client.list_collections`` не вызывается, тк base build стабится.

Проверяем артефакты: ``summary.json`` + ``metrics_matrix.md``, наличие записей
по каждому реранкеру.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def test_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from core.rag.rerankers.reranker_base_types import RerankRequest, RerankResult
    from core.rag.retrievers.retriever_base_types import (
        RetrievalRequest,
        RetrievalResult,
        RetrievedDoc,
    )
    from eval.retrieval import run_final_reranker_matrix as f7

    class _Base:
        name = "base_stub"

        def search(self, request: RetrievalRequest) -> RetrievalResult:
            return RetrievalResult(
                docs=[
                    RetrievedDoc(text="рефлексия в учебнике", score=0.9, rank=1, chunk_id="A"),
                    RetrievedDoc(text="нерелевантный текст", score=0.1, rank=2, chunk_id="B"),
                ]
            )

    class _CannedReranker:
        name = "stub_cross"

        def __init__(self, score_by_cid):
            self._s = score_by_cid

        def rerank(self, request: RerankRequest) -> RerankResult:
            scored = sorted(request.docs, key=lambda d: self._s.get(d.chunk_id, 0.0), reverse=True)
            return RerankResult(docs=scored[: request.top_k], timings_ms=0)

    monkeypatch.setattr(
        f7, "_build_base_retriever",
        lambda **kw: (_Base(), {"kind": "dense", "base_id": "full_mpnet", "collection": "x"}),
    )
    monkeypatch.setattr(
        f7, "chroma_http_client",
        lambda **kw: type("C", (), {"list_collections": lambda self: []})(),
    )
    monkeypatch.setattr(
        f7, "build_reranker",
        lambda name, mode="auto", device=None: _CannedReranker({"A": 0.9, "B": 0.1}),
    )

    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "что такое рефлексия?", "references": ["рефлексия"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(f7, "_ROOT", tmp_path)

    argv = [
        "prog",
        "--base", "dense:full_mpnet",
        "--rerankers", "fake-A,fake-B",
        "--dense-registry", str(REPO / "eval" / "config" / "embedding_bases_full_corpus.yaml"),
        "--eval-file", str(eval_file),
        "--candidate-top-k", "2",
        "--top-k", "2",
        "--k", "1,2",
        "--metric-modes", "substring",
        "--include-none-baseline",
        "--max-rows", "1",
        "--seed", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = f7.main()
    assert rc == 0

    runs_root = tmp_path / "eval" / "runs"
    run_dir = sorted(runs_root.iterdir())[-1]
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics_matrix.md").is_file()
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_keys = {"__no_rerank__", "fake-A", "fake-B"}
    assert set(summ["results"].keys()) == expected_keys
    for k in expected_keys:
        cell = summ["results"][k]
        assert cell["ok"] is True
        assert cell["metrics"]["recall@1"] == pytest.approx(1.0)

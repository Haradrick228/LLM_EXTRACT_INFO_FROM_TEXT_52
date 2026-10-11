"""Smoke-тест ``eval.retrieval.run_query_transform_sweep``.

Не ходит в реальную Chroma/Ollama:

- ``chroma_http_client`` / ``build_chroma_embedding_function`` / ``_build_dense_retriever``
  заменяются стабами;
- ``_build_llm_client`` возвращает фиксированный фейк-LLM;
- ``SparseBm25Retriever.from_dir`` подменяется заглушкой.

Проверяем, что:

- запуск отрабатывает без ошибок;
- получены 4 job'а: dense_only, hyde, mqr, hyde_mqr_dense_bm25_rrf;
- summary.json и metrics_matrix.md созданы;
- HyDE использует гипотетический текст (виден в trace.debug).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def test_run_query_transform_sweep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from core.rag.retrievers.retriever_base_types import (
        RetrievalRequest,
        RetrievalResult,
        RetrievedDoc,
    )
    from eval.retrieval import run_query_transform_sweep as sweep

    class _DummyDense:
        name = "dense_dummy"

        def search(self, request: RetrievalRequest) -> RetrievalResult:
            txt = "релевантный текст рефлексия " + request.query[:32]
            return RetrievalResult(
                docs=[
                    RetrievedDoc(text=txt, score=0.9, rank=1, chunk_id="C1"),
                    RetrievedDoc(text="шум", score=0.1, rank=2, chunk_id="C2"),
                ]
            )

    class _DummyBm25:
        name = "bm25_corpus"

        def search(self, request: RetrievalRequest) -> RetrievalResult:
            return RetrievalResult(
                docs=[RetrievedDoc(text="бм25 чанк рефлексия", score=0.5, rank=1, chunk_id="B1")]
            )

    class _FakeLLM:
        def complete(self, prompt: str, temperature=None, max_tokens=None) -> str:
            if "перефразировки" in prompt.lower():
                return "1. альтернатива А\n2. альтернатива Б\n3. альтернатива В\n4. альтернатива Г\n"
            return "Гипотетический ответ про рефлексию — это умение."

    chroma_names = {"corpus_phase_b_mpnet"}

    class _DummyChromaClient:
        def list_collections(self):
            return [type("C", (), {"name": n})() for n in chroma_names]

        def get_collection(self, name, embedding_function=None):
            return object()

    monkeypatch.setattr(sweep, "chroma_http_client", lambda **kw: _DummyChromaClient())
    monkeypatch.setattr(sweep, "build_chroma_embedding_function", lambda spec: None)
    monkeypatch.setattr(sweep, "_build_dense_retriever", lambda *a, **kw: _DummyDense())
    monkeypatch.setattr(sweep, "_build_llm_client", lambda provider, model: _FakeLLM())
    monkeypatch.setattr(
        sweep.SparseBm25Retriever, "from_dir",
        classmethod(lambda cls, p, name="bm25_corpus": _DummyBm25()),
    )

    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "что такое рефлексия?", "references": ["рефлексия"]}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    # bm25 dir must exist
    bm25_dir = tmp_path / "bm25"
    bm25_dir.mkdir()

    monkeypatch.setattr(sweep, "_ROOT", tmp_path)

    argv = [
        "prog",
        "--dense-registry", str(REPO / "eval" / "config" / "embedding_bases_full_corpus.yaml"),
        "--scenarios", "full_mpnet",
        "--bm25-dir", str(bm25_dir),
        "--eval-file", str(eval_file),
        "--fetch-k", "5",
        "--k", "1,3",
        "--rrf-k", "60",
        "--metric-modes", "substring",
        "--mqr-n-queries", "2",
        "--max-rows", "1",
        "--llm-provider", "ollama",
        "--llm-model", "fake-model",
        "--seed", "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = sweep.main()
    assert rc == 0

    runs_root = tmp_path / "eval" / "runs"
    run_dir = sorted(runs_root.iterdir())[-1]
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "metrics_matrix.md").is_file()

    summ: Dict[str, Any] = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    bucket = summ["results"]["full_mpnet"]
    for job in ("dense_only", "hyde", "mqr", "hyde_mqr_dense_bm25_rrf"):
        assert bucket[job]["ok"] is True, f"job {job} failed: {bucket[job]}"
        assert bucket[job]["metrics"]["recall@1"] == pytest.approx(1.0)

    # HyDE trace must record hypothetical text was used.
    hyde_trace = run_dir / "full_mpnet" / "hyde" / "hyde.trace.jsonl"
    line = [l for l in hyde_trace.read_text(encoding="utf-8").splitlines() if l][0]
    payload = json.loads(line)
    assert payload["retrieval_debug"]["hyde_used_query_is_hypothetical"] is True
    assert "Гипотетический" in payload["retrieval_debug"]["hyde_hypothetical_text"]

    # MQR trace must show >=2 rewrites generated.
    mqr_trace = run_dir / "full_mpnet" / "mqr" / "mqr.trace.jsonl"
    line = [l for l in mqr_trace.read_text(encoding="utf-8").splitlines() if l][0]
    payload = json.loads(line)
    assert payload["retrieval_debug"]["mqr_n_rewrites_generated"] == 2
    assert payload["retrieval_debug"]["mqr_include_original_query"] is True

"""Deprecated namespace.

Содержимое ``eval/phase_b/`` переехало по новой схеме:

| Старое имя | Новое имя |
|---|---|
| ``eval.phase_b.run_full_corpus_rerank_matrix`` | :mod:`eval.retrieval.run_retrieval_matrix` |
| ``eval.phase_b.index_jsonl_bases``             | :mod:`eval.indexing.index_corpus_from_jsonl` |
| ``eval.phase_b.index_gold_fixture_bases``      | :mod:`eval.indexing.index_gold_fixture_collections` |
| ``eval.phase_b.verify_corpus_pipeline``        | :mod:`eval.indexing.verify_corpus_indexing_pipeline` |
| ``eval.phase_b.verify_collections``            | :mod:`eval.indexing.verify_collections` |
| ``eval.phase_b.chroma_embed_factory``          | :mod:`eval.indexing.chroma_embedding_function_factory` |
| ``eval.phase_b.jsonl_corpus_stream``           | :mod:`eval.indexing.jsonl_corpus_stream` |
| ``eval.phase_b.fixture_chunks``                | :mod:`eval.datasets.load_gold_fixture_chunks` |
| ``eval.phase_b.registry``                      | :mod:`eval.experiments.phase_b_collections_registry` |
| ``eval.phase_b.embedding_bases.yaml``          | ``eval/config/embedding_bases_fixture.yaml`` |
| ``eval.phase_b.embedding_bases_full_corpus.yaml`` | ``eval/config/embedding_bases_full_corpus.yaml`` |

Импорты из ``eval.phase_b.*`` продолжают работать через shim ниже, но при
этом печатают ``DeprecationWarning``. Новый код должен использовать
финальные пути.
"""
from __future__ import annotations

import importlib
import sys
import warnings

# Карта подмодулей → новые пути. Используется ленивым __getattr__ ниже,
# чтобы не импортировать всё подряд при загрузке пакета.
_REEXPORT_MAP = {
    "run_full_corpus_rerank_matrix": "eval.retrieval.run_retrieval_matrix",
    "index_jsonl_bases":            "eval.indexing.index_corpus_from_jsonl",
    "index_gold_fixture_bases":     "eval.indexing.index_gold_fixture_collections",
    "verify_corpus_pipeline":       "eval.indexing.verify_corpus_indexing_pipeline",
    "verify_collections":           "eval.indexing.verify_collections",
    "chroma_embed_factory":         "eval.indexing.chroma_embedding_function_factory",
    "jsonl_corpus_stream":          "eval.indexing.jsonl_corpus_stream",
    "fixture_chunks":               "eval.datasets.load_gold_fixture_chunks",
    "registry":                     "eval.experiments.phase_b_collections_registry",
}


def __getattr__(name: str):
    target = _REEXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'eval.phase_b' has no attribute {name!r}")
    warnings.warn(
        f"eval.phase_b.{name} is deprecated; use {target}",
        DeprecationWarning,
        stacklevel=2,
    )
    module = importlib.import_module(target)
    sys.modules[f"{__name__}.{name}"] = module
    return module


__all__ = sorted(_REEXPORT_MAP)

# `core/rag/` — компоненты RAG-стека

Все компоненты разделены по обязанностям, чтобы матрица `retriever × reranker × chunker`
собиралась без хождений по дереву.

## Дерево

```
core/rag/
├── retrievers/             # как достаём кандидатов
│   ├── retriever_base_types.py
│   ├── dense_chroma_retriever.py            # F2-: dense через ChromaDB (плановое имя)
│   ├── sparse_bm25_retriever.py             # F2:  BM25 sidecar
│   ├── hybrid_rrf_retriever.py              # F2:  RRF-fusion
│   ├── hybrid_linear_weighted_retriever.py  # F2:  linear/weighted fusion
│   ├── bge_m3_multi_vector_retriever.py     # F4:  dense+sparse+ColBERT
│   ├── hyde_query_expansion_retriever.py    # F3:  HyDE
│   └── multi_query_rewrite_retriever.py     # F3:  Multi-Query Rewriting
│
├── rerankers/              # как переставляем top-fetch_k
│   ├── reranker_base_types.py
│   ├── cross_encoder_reranker.py            # CrossEncoder / bi-encoder (бывший rerank_registry)
│   └── colbert_late_interaction_reranker.py # F4
│
├── chunkers/               # как режем документ при индексации
│   ├── chunker_base_types.py
│   ├── fixed_size_chunker.py                # F5: 650/250
│   ├── parent_child_chunker.py              # F5: child 256/64 + parent 1024/128
│   └── semantic_boundary_chunker.py         # F5: эмбеддинг-границы
│
├── metrics/                # как считаем "релевантно ли"
│   ├── substring_match_hit.py
│   ├── token_overlap_f1_hit.py              # F1
│   ├── semantic_similarity_hit.py           # F1
│   ├── source_id_match_hit.py               # F1
│   └── ranking_aggregates.py                # recall@k, precision@k, MRR, nDCG@k
│
├── service/                # сборка end-to-end RAG
│   ├── rag_service.py                       # основной сервис
│   ├── rag_service_legacy_facade.py         # обёртка для внешнего API
│   └── llm_response_query_cache.py          # кэш ответов LLM в Postgres
│
└── llm/                    # LLM-клиенты (используются `service/`)
    └── deepseek_llm_client.py
```

## Принципы

1. **Один файл — одна обязанность**. Если новый ретривер — это новый файл в
   `retrievers/`, а не +200 строк в `rag_service.py`.
2. **Контракты — в `*_base_types.py`**. Любой ретривер реализует
   `Retriever.search(...)`, любой реранкер — `Reranker.rerank(...)`. Это даёт
   `HybridRrfRetriever` собирать вообще разнородные источники.
3. **Совместимость со старыми импортами** держится shim‑модулями
   (`core/rag/rerank_registry.py`, `core/rag/rag_service.py` и т.п.) — они
   печатают `DeprecationWarning` и re-export'ят символы. Новый код должен
   импортировать напрямую: `from core.rag.rerankers.cross_encoder_reranker import build_reranker`.
4. **Логирование** — через `common.structured_logging.get_logger(__name__)`,
   тайминги — через `with timed(logger, "dense.query"): ...`.

## Что НЕ лежит в `core/rag/`

- Eval-сценарии и CLI: `eval/retrieval/`, `eval/indexing/`, `eval/datasets/`.
- Конфиги retrieval: `eval/config/*.yaml`.
- GraphRAG (graph-based context): `core/graphrag/` — отдельный слой, ниже
  внешнего API; RAG-сервис при необходимости делегирует.

## Итоговые таблицы retrieval (Phase B, full corpus)

Сводные метрики по матрицам F2–F7 и заметки по B3 лежат в `eval/docs/`: [eval/docs/RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md](../../eval/docs/RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md); диагноз низкого recall — [eval/docs/RETRIEVAL_DIAGNOSIS_2026_05.md](../../eval/docs/RETRIEVAL_DIAGNOSIS_2026_05.md).

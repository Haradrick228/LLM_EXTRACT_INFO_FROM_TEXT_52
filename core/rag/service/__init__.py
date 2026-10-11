"""Сервисный слой RAG.

- :mod:`core.rag.service.rag_service` — основной сервис: graph context →
  fallback к vector retrieval → LLM → кэш ответов;
- :mod:`core.rag.service.rag_service_legacy_facade` — обёртка для старого
  внешнего API (бывший ``rag_service_wrapper``);
- :mod:`core.rag.service.llm_response_query_cache` — кэш ответов LLM в Postgres
  (бывший ``query_cache``); кэш именно ответов LLM, не самих запросов к ретриверу.
"""

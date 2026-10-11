"""Индексация корпуса в Chroma и sidecar-индексы.

CLI:

- :mod:`eval.indexing.index_corpus_from_jsonl`           — полный корпус из JSONL;
- :mod:`eval.indexing.index_gold_fixture_collections`    — фикстура из 101 чанков;
- :mod:`eval.indexing.export_chroma_collection_to_jsonl` — экспорт коллекции;
- :mod:`eval.indexing.verify_corpus_indexing_pipeline`   — проверка count/dim;
- :mod:`eval.indexing.reindex_with_text_cleaning`        — переиндексация с очисткой.

Вспомогательные модули:

- :mod:`eval.indexing.chroma_embedding_function_factory` — фабрика
  ``embedding_function`` по имени HuggingFace-модели;
- :mod:`eval.indexing.jsonl_corpus_stream`               — потоковая загрузка чанков из JSONL.
"""

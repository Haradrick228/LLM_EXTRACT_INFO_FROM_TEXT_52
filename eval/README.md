# `eval/` — оценка retrieval и качества ответов

После рефакторинга F0 модуль разбит по функциональным подкаталогам.
Старые пути работают через shim‑модули с `DeprecationWarning`.

## Дерево

```
eval/
├── retrieval/               # CLI и оркестраторы метрик (recall@k, MRR, nDCG@k, answer score)
│   ├── run_retrieval_eval.py        # одиночный прогон (одна коллекция × один реранкер)
│   ├── run_retrieval_matrix.py      # матрица сценариев на полном корпусе (бывший run_full_corpus_rerank_matrix)
│   ├── run_profile_metrics.py       # матрица профилей из eval/config/rag_profiles.yaml
│   ├── run_eval_with_profile.py     # разовый запуск под профилем
│   └── run_answer_quality_eval.py   # answer-eval (бывший run_answer_eval)
│
├── indexing/                # индексация корпуса в Chroma и sidecar
│   ├── index_corpus_from_jsonl.py
│   ├── index_gold_fixture_collections.py
│   ├── export_chroma_collection_to_jsonl.py
│   ├── verify_corpus_indexing_pipeline.py
│   ├── verify_collections.py
│   ├── reindex_with_text_cleaning.py
│   ├── chroma_embedding_function_factory.py
│   └── jsonl_corpus_stream.py
│
├── datasets/                # gold-датасеты и инструменты их сборки
│   ├── eval_set_gold_v2.jsonl
│   ├── eval_set_gold_v2.meta.json
│   ├── eval_set_with_answer_ref.jsonl
│   ├── eval_set_with_answer_ref_sample20.jsonl
│   ├── eval_set.jsonl
│   ├── chunks_sample.jsonl
│   ├── materialize_gold_v2_from_answer_ref.py
│   ├── build_gold_v2_from_chroma_chunks.py
│   ├── generate_synthetic_gold_v2.py
│   ├── validate_gold_v2_schema.py
│   └── load_gold_fixture_chunks.py
│
├── reporting/               # форматирование артефактов eval-прогонов
│   └── log_eval_run_config_banner.py
│
├── diagnostics/             # worst-trace, funnel A–D, semantic probe, answer-in-run-dir, LLM relevance, neg-control
│   ├── metrics_ladder_funnel.py
│   ├── semantic_threshold_probe.py
│   ├── run_answer_eval_in_run_dir.py
│   ├── summarize_worst_retrieval_traces.py
│   ├── shuffle_eval_questions_neg_control.py
│   └── llm_chunk_relevance_deepseek.py
│
├── experiments/             # реестр replay-экспериментов
│   ├── replay_experiment.py
│   └── phase_b_collections_registry.py
│
├── config/                  # YAML-конфиги retrieval-стека
│   ├── embedding_bases_fixture.yaml
│   ├── embedding_bases_full_corpus.yaml
│   ├── rag_profiles.yaml
│   └── experiments_registry.yaml
│
├── phase_b/                 # deprecated shim для legacy импортов (eval.phase_b.*)
├── chroma_utils.py          # клиент Chroma (общий)
├── text_pipeline.py         # NFKC/whitespace-нормализация
├── profile_env.py           # сборка env для профиля
└── runs/                    # immutable артефакты прогонов (создаются автоматически)
```

## Датасеты (gold v2)

- **`datasets/eval_set_gold_v2.jsonl`** + **`datasets/eval_set_gold_v2.meta.json`** — golden v2.
- **`datasets/eval_set_with_answer_ref.jsonl`** — 101 вопрос с `answer_ref` (основной источник для materialize).
- **`datasets/eval_set_with_answer_ref_sample20.jsonl`** — 20 вопросов для быстрых прогонов.
- **`datasets/chunks_sample.jsonl`** — чанки, **ровно те**, что соответствуют строкам gold v2 (после материализации).

Сборка:

```bash
# Основной путь без LLM:
python -m eval.datasets.materialize_gold_v2_from_answer_ref --n 100 --yes
# Альтернатива с Chroma + Ollama:
python -m eval.datasets.build_gold_v2_from_chroma_chunks --collection default --n 50 --yes
# Только синтетика для CI:
python -m eval.datasets.generate_synthetic_gold_v2 --n 100
# Проверка целостности:
python -m eval.datasets.validate_gold_v2_schema
```

## Лестница метрик (A–D, Wilson, conditional)

См. **[docs/METRICS_LADDER.md](docs/METRICS_LADDER.md)** и CLI `python -m eval.diagnostics.metrics_ladder_funnel`. Сводный отчёт исследования Phase B (валидность метрик + B2/B3 + replay): **[docs/PHASE_B_RESEARCH_METRICS_REPORT.md](docs/PHASE_B_RESEARCH_METRICS_REPORT.md)**. Индекс заметок: **[docs/README.md](docs/README.md)**.

## Запуск с хоста Windows (Docker уже поднят)

В [`.env`](../.env) для docker-compose часто задано `CHROMA_HOST=chroma` и порт `8000` — это имя сервиса внутри сети compose. Если `uvicorn`/`RagService` запускаете на машине, а не в контейнере `app`, выставьте для Chroma:

- `CHROMA_HOST=127.0.0.1`
- `CHROMA_PORT=18000` (проброс из `infra/docker-compose.yml`: `18000:8000`)

Если в `.env` для compose остаётся `CHROMA_HOST=chroma`, для **eval с хоста** задайте `RAG_EVAL_CHROMA_HOST=127.0.0.1`, `RAG_EVAL_CHROMA_PORT=18000` (их читают `run_retrieval_eval`, `run_answer_quality_eval`, `run_profile_metrics`, `run_retrieval_matrix`).

Ollama в Docker — на `127.0.0.1:11434`; для LLM: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`.

Быстрая проверка без PyTorch/HF: `python scripts/verify_local_stack_running.py`.

Интеграционные pytest (нужен поднятый Chroma): `CHROMA_INTEGRATION=1 python -m pytest tests/integration`.

## Запуск метрик

```bash
# Одиночный прогон (одна коллекция × один реранкер):
python -m eval.retrieval.run_retrieval_eval \
    --eval-file eval/datasets/eval_set_gold_v2.jsonl \
    --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --fetch-k 40 --k 5,10 \
    --rerank-model BAAI/bge-reranker-v2-m3 --rerank-mode cross

# Матрица B3 (5 dense × N реранкеров) на полном корпусе:
python -m eval.retrieval.run_retrieval_matrix --expected-rows 61543

# Replay эксперимента по реестру:
python -m eval.experiments.replay_experiment list
python -m eval.experiments.replay_experiment run retrieval_matrix_gold_v2_fixture
```

## Сводка прогонов full corpus (2026-05)

- Таблицы **F2 / F3 / F4 / F7** (и контекст **B3**): [docs/RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md](docs/RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md) (цифры из [analysis/reports/retrieval_recovery_2026_05.html](../analysis/reports/retrieval_recovery_2026_05.html)).
- Разбор низких метрик, логов/трейсов и гипотез: [docs/RETRIEVAL_DIAGNOSIS_2026_05.md](docs/RETRIEVAL_DIAGNOSIS_2026_05.md).
- Канонический HTML-отчёт: [analysis/reports/final_canonical_experiment_2026_05.html](../analysis/reports/final_canonical_experiment_2026_05.html).

## Размеры корпуса (важно при чтении метрик)

| Контекст                                | n_chunks       | dense `recall@10` (mpnet) |
|---|---|---|
| **fixture** (`gold_v2_fixture`)         | 101            | ~0.6                       |
| **full corpus** (`corpus_phase_b_*`)    | 61 543         | 0.03–0.09 (bm25/hybrid поднимают до 0.2–0.4) |

Метрики на fixture исторически были «лучше» именно потому, что в коллекции лежали ровно те чанки, по которым составлен gold. Реальная задача — full corpus; туда направлена матрица B3 и план улучшений в `retrieval_recovery_plan_*.plan.md`.

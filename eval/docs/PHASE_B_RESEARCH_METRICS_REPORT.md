# Phase B — финальный отчёт по метрикам и full-corpus B3

**Полный финальный отчёт (для людей):**  
[FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md](FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md) ·  
[HTML](../analysis/reports/final_canonical_experiment_2026_05.html) — **v2**: вкладки, RAG-пайплайн, словарь метрик, **вкладка «Рекомендация»** (лидеры по метрикам / этапам), answer-eval.

Каноническая ячейка для **Phase B gate** — `substring::recall@10` (сейчас ничья FRIDA+BGE / FRIDA+Jina). Альтернативы по token_f1 / semantic / avg_ref_sim — в разделе 2 человекочитаемого отчёта.

*Этот файл — инженерный справочник по командам; для презентации используйте ссылку выше.*

Документ фиксирует **цепочку доверия** к метрикам и **воспроизводимые** команды. Канон интерпретации: [METRICS_LADDER.md](METRICS_LADDER.md).

**Протокол:** [config/final_research_protocol.yaml](config/final_research_protocol.yaml) · **Валидность:** `eval/runs/final_research_validity_suite.json` · **Пайплайн:** `scripts/run_final_research_full_corpus.ps1`

## 1. Доказательство валидности метрик (до таблицы 6×3)

| Проверка | Команда / артефакт | Ожидание |
|----------|-------------------|----------|
| Негативный контроль | `eval.diagnostics.shuffle_eval_questions_neg_control` + `run_retrieval_eval` на shuffled JSONL | Метрики заметно ниже честного gold |
| Внешний sanity | Отдельный прогон на SQuAD / простой коллекции | Высокий сигнал там, где ожидается |
| B2 vs B3 | B2: `EVAL_SCENARIO_TYPE=fixture_smoke`, ~101 чанк; B3: `full_corpus`, ~61k | На фикстуре recall выше; **не смешивать** строки одной таблицы |
| Лестница A–D | `python -m eval.diagnostics.metrics_ladder_funnel ...` | Доля A объясняет потолок substring; `conditional_given_oracle_pool` — ранг при gold в пуле |
| Wilson CI | В выходе `metrics_ladder_funnel` | Честный интервал на n=101 |
| Semantic bundle | `_meta.metric_modes` включает `semantic`, зафиксированы `semantic_threshold` / `semantic_model` | Сравнение прогонов без «разных порогов» |
| Пост-обработка одним входом | `python -m eval.diagnostics.run_post_eval_ladder --trace ... --ladder-out-json ... --probe-out-json ...` | Один CLI для ladder + probe |

### Ограничения gold v2

Текущий `eval/datasets/eval_set_gold_v2.jsonl` **не** содержит стабильных `chunk_document_id` из индексируемого корпуса — релевантность задаётся текстом `references` / `reference_quote`. Строгий BEIR-style hit по id требует пересборки gold (см. METRICS_LADDER).

## 2. B3 — полная матрица dense × rerank (полный корпус)

- Оркестратор: `python -m eval.retrieval.run_retrieval_matrix` (алиас `eval.phase_b.run_full_corpus_rerank_matrix`).
- По умолчанию: **`--metric-modes substring,token_f1,semantic`**, `RETRIEVAL_EVAL_SHUFFLE=0`, `EVAL_SCENARIO_TYPE=full_corpus` в subprocess.
- Реестр: `python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix` — в `env` заданы `B3_BASE_IDS`, `RETRIEVAL_METRIC_MODES`, `EVAL_SCENARIO_TYPE`.
- Артефакты прогона: `eval/runs/<ts>_b3_full_corpus_rerank/matrix_manifest.json`, `summary.json`, `metrics_matrix.md` (таблицы по **всем** числовым ключам из `retrieval_*.json`, включая `token_f1::` / `semantic::`).

### Answer-eval по ячейке матрицы

Мета лежит в `<run_dir>/<base_id>/retrieval_<profile>.json`, а не в корне `run_dir`:

```bash
python -m eval.diagnostics.run_answer_eval_in_run_dir ^
  --run-dir eval/runs/<ts>_b3_full_corpus_rerank/full_bge_m3 ^
  --retrieval-json eval/runs/<ts>_b3_full_corpus_rerank/full_bge_m3/retrieval_rerank_jina_v3.json ^
  --max-rows 0
```

Скорер сходства ответа (`ANSWER_REF_EMBED_MODEL`, по умолчанию MiniLM) **отличается** от query-embedder коллекции — это осознанный контракт; `RERANK_*` подтягиваются из `_meta` retrieval JSON.

## 3. B2 — фикстура (слоты `phase_b_dense_slot_*`)

`python -m eval.replay_experiment run phase_b_dense_slot_2_rerank_matrix` и т.д. В `env`: `EVAL_SCENARIO_TYPE=fixture_smoke`, `RETRIEVAL_METRIC_MODES`, `RETRIEVAL_EVAL_SHUFFLE=0`. Слот 5/6: флаг `--embed-trust-remote-code` в команде реестра.

## 4. F1 / F2 / F3 / F4 после baseline

- **F1** (`f1_retrieval_eval_with_new_metrics`): в команде явно `substring,token_f1,semantic` и semantic model/threshold; в `env` — `EVAL_SCENARIO_TYPE`, `RETRIEVAL_METRIC_MODES`.
- **F2 / F3 / F4**: запускать **после** зафиксированного B3; при расширении sweep поддерживать тот же контракт `_meta` (см. исходники соответствующих `run_*`).

## 5. Исправления кода (ответ на расхождение Chroma / RagService)

- `RagService` принимает **`chroma_embed_model`** в kwargs и выставляет коллекцию/эмбеддер явно.
- `run_answer_quality_eval` делает **lazy-import** `RagService` после выравнивания `os.environ` и передаёт `chroma_collection`, `chroma_host`, `chroma_port`, `chroma_embed_model` в конструктор — устраняет использование «замороженных» дефолтов `ChromaConfig` с момента первого импорта модуля.

## 6. См. также

- [PHASE_B_SCOPE.md](PHASE_B_SCOPE.md)
- [PHASE_B3_FULL_CORPUS_RERANK_PLAN.md](PHASE_B3_FULL_CORPUS_RERANK_PLAN.md)
- [config/experiments_registry.yaml](config/experiments_registry.yaml)

<!-- FINAL_RESEARCH_AUTO_START -->

## Авто-сводка (protocol `2026-05-27`)

> **Только канонический прогон** — таблицы ниже из одного `run_dir`; исторические B3 (fetch_k=100 и частичные) **не включены**.

- **Каноническая B3 6×3:** `eval\runs\20260527_161527_b3_full_corpus_rerank`
- **fetch_k:** `40` · **metric_modes:** `substring,token_f1,semantic` · **вопросов:** 101 (gold v2)
- **Протокол:** `eval/config/final_research_protocol.yaml`
- **Replay B3:** `python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix`
- **Validity suite:** `python -m eval.diagnostics.run_final_research_validity_suite`

### Валидность метрик (pass/fail)

| Проверка | Статус | Детали |
|---|---|---|
| external_sanity | PASS | 0.825 |
| neg_control | PASS | honest=0.0644, shuffled=0.0297, gap=0.0347 |
| metrics_ladder | PASS | A=0.594059405940594 |
| semantic_threshold_probe | PASS | eval\runs\20260513_205851_gold_on_full_corpus_bge_m3\semantic_threshold_probe.json |

Neg-control honest cell: `eval\runs\20260527_161527_b3_full_corpus_rerank\full_frida\retrieval_rerank_jina_v3.json` (full_frida × rerank_jina_v3).

### Лестница A–D (anchor trace, `eval/runs/20260513_205851_gold_on_full_corpus_bge_m3`)

| Показатель | Значение |
|---|---|
| class A (ref в корпусе) | 0.5941 |
| class D (miss@k) | 0.0594 |
| Wilson CI recall@10 (post-rank) | 0.0275 – 0.1236 (n=6) |
| conditional_given_oracle_pool (mean recall@k) | 0.5625 (n=8) |

### Answer-layer (top retrieval по `substring::recall@10`, канонический B3)

| rank | scenario | profile | retrieval RC | note |
|---|---|---|---|---|
| 1 | full_frida | rerank_jina_v3 | 0 | OK |
| 2 | full_frida | rerank_bge_v2_m3 | 0 | OK |
| 3 | full_frida | baseline_mpnet_rerank | 0 | OK |

### Leaderboard — `substring::recall@10` (канонический B3)

| rank | dense base | rerank profile | recall@10 |
|---|---|---|---|
| 1 | FRIDA (FRIDA) | rerank_jina_v3 | 0.0644 |
| 2 | FRIDA (FRIDA) | rerank_bge_v2_m3 | 0.0644 |
| 3 | Jina-embeddings-v3 (jina-embeddings-v3) | rerank_bge_v2_m3 | 0.0594 |
| 4 | E5-large-instruct (multilingual-e5-large-instruct) | rerank_jina_v3 | 0.0594 |
| 5 | E5-large-instruct (multilingual-e5-large-instruct) | rerank_bge_v2_m3 | 0.0594 |
| 6 | BGE-M3 (bge-m3) | rerank_bge_v2_m3 | 0.0545 |
| 7 | BGE-M3 (bge-m3) | rerank_jina_v3 | 0.0446 |
| 8 | FRIDA (FRIDA) | baseline_mpnet_rerank | 0.0396 |

### Матрица 6×3 — все метрики (32 таблиц)

По одной таблице на каждый числовой ключ из `retrieval_*.json` ячеек канонического прогона.

#### `recall@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0050 | 0.0198 | 0.0198 |
| BGE-M3 (bge-m3) | 0.0248 | 0.0545 | 0.0446 |
| FRIDA (FRIDA) | 0.0396 | 0.0644 | 0.0644 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0594 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0149 | 0.0594 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0050 | 0.0198 | 0.0198 |

#### `recall@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0099 | 0.0198 | 0.0198 |
| BGE-M3 (bge-m3) | 0.0248 | 0.0545 | 0.0446 |
| FRIDA (FRIDA) | 0.0545 | 0.0743 | 0.0743 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0297 | 0.0693 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0198 | 0.0594 | 0.0396 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0099 | 0.0198 | 0.0198 |

#### `recall@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0198 | 0.0099 |
| BGE-M3 (bge-m3) | 0.0099 | 0.0248 | 0.0347 |
| FRIDA (FRIDA) | 0.0000 | 0.0297 | 0.0594 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0297 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0297 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0198 | 0.0099 |

#### `mrr`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0025 | 0.0297 | 0.0142 |
| BGE-M3 (bge-m3) | 0.0086 | 0.0209 | 0.0442 |
| FRIDA (FRIDA) | 0.0072 | 0.0236 | 0.0401 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0132 | 0.0224 | 0.0520 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0041 | 0.0208 | 0.0336 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0025 | 0.0297 | 0.0142 |

#### `ndcg@3`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0297 | 0.0091 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0061 | 0.0319 |
| FRIDA (FRIDA) | 0.0000 | 0.0061 | 0.0259 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0061 | 0.0061 | 0.0513 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0061 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0297 | 0.0091 |

#### `precision@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0109 | 0.0040 |
| BGE-M3 (bge-m3) | 0.0109 | 0.0158 | 0.0228 |
| FRIDA (FRIDA) | 0.0059 | 0.0218 | 0.0248 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0188 | 0.0109 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0030 | 0.0198 | 0.0089 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0109 | 0.0040 |

#### `precision@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0099 | 0.0054 |
| BGE-M3 (bge-m3) | 0.0124 | 0.0208 | 0.0178 |
| FRIDA (FRIDA) | 0.0109 | 0.0272 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0084 | 0.0203 | 0.0149 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0054 | 0.0168 | 0.0069 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0099 | 0.0054 |

#### `precision@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0178 | 0.0059 |
| BGE-M3 (bge-m3) | 0.0059 | 0.0099 | 0.0198 |
| FRIDA (FRIDA) | 0.0000 | 0.0099 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0040 | 0.0139 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0119 | 0.0158 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0178 | 0.0059 |

#### `semantic::mrr`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3830 | 0.5692 | 0.4920 |
| BGE-M3 (bge-m3) | 0.3725 | 0.6079 | 0.5281 |
| FRIDA (FRIDA) | 0.3402 | 0.5278 | 0.4814 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.3428 | 0.5831 | 0.4587 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.4289 | 0.6157 | 0.5285 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3784 | 0.5668 | 0.5029 |

#### `semantic::ndcg@3`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3763 | 0.5639 | 0.4886 |
| BGE-M3 (bge-m3) | 0.3795 | 0.6037 | 0.5273 |
| FRIDA (FRIDA) | 0.3270 | 0.5392 | 0.4966 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.3395 | 0.6035 | 0.4803 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.4389 | 0.6012 | 0.5491 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3677 | 0.5540 | 0.5034 |

#### `semantic::precision@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3376 | 0.4178 | 0.3752 |
| BGE-M3 (bge-m3) | 0.3089 | 0.4020 | 0.3604 |
| FRIDA (FRIDA) | 0.2604 | 0.3446 | 0.3079 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2634 | 0.3693 | 0.3020 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3267 | 0.4089 | 0.3762 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3376 | 0.4149 | 0.3772 |

#### `semantic::precision@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3391 | 0.3748 | 0.3619 |
| BGE-M3 (bge-m3) | 0.3089 | 0.3441 | 0.3322 |
| FRIDA (FRIDA) | 0.2535 | 0.2965 | 0.2748 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2688 | 0.3203 | 0.2777 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3203 | 0.3609 | 0.3485 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3396 | 0.3762 | 0.3609 |

#### `semantic::precision@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3129 | 0.4475 | 0.3980 |
| BGE-M3 (bge-m3) | 0.2990 | 0.4733 | 0.3921 |
| FRIDA (FRIDA) | 0.2475 | 0.4040 | 0.3525 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2495 | 0.4455 | 0.3307 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3327 | 0.4752 | 0.4059 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3129 | 0.4436 | 0.4040 |

#### `semantic::recall@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.5792 | 0.6634 | 0.6089 |
| BGE-M3 (bge-m3) | 0.6287 | 0.6733 | 0.6188 |
| FRIDA (FRIDA) | 0.5594 | 0.6040 | 0.6238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5743 | 0.6634 | 0.6040 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.6139 | 0.7129 | 0.6881 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.5891 | 0.6535 | 0.6188 |

#### `semantic::recall@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.6436 | 0.7178 | 0.6782 |
| BGE-M3 (bge-m3) | 0.6832 | 0.7129 | 0.6782 |
| FRIDA (FRIDA) | 0.6188 | 0.6535 | 0.6634 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.6782 | 0.7079 | 0.6832 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.6931 | 0.7673 | 0.7277 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.6634 | 0.7178 | 0.6881 |

#### `semantic::recall@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.4901 | 0.6040 | 0.5644 |
| BGE-M3 (bge-m3) | 0.5000 | 0.6337 | 0.5693 |
| FRIDA (FRIDA) | 0.4604 | 0.5594 | 0.5693 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.4505 | 0.6287 | 0.5396 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.5446 | 0.6634 | 0.6386 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.4901 | 0.6040 | 0.5743 |

#### `substring::mrr`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0025 | 0.0297 | 0.0142 |
| BGE-M3 (bge-m3) | 0.0086 | 0.0209 | 0.0442 |
| FRIDA (FRIDA) | 0.0072 | 0.0236 | 0.0401 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0132 | 0.0224 | 0.0520 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0041 | 0.0208 | 0.0336 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0025 | 0.0297 | 0.0142 |

#### `substring::ndcg@3`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0297 | 0.0091 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0061 | 0.0319 |
| FRIDA (FRIDA) | 0.0000 | 0.0061 | 0.0259 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0061 | 0.0061 | 0.0513 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0061 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0297 | 0.0091 |

#### `substring::precision@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0109 | 0.0040 |
| BGE-M3 (bge-m3) | 0.0109 | 0.0158 | 0.0228 |
| FRIDA (FRIDA) | 0.0059 | 0.0218 | 0.0248 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0188 | 0.0109 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0030 | 0.0198 | 0.0089 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0109 | 0.0040 |

#### `substring::precision@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0099 | 0.0054 |
| BGE-M3 (bge-m3) | 0.0124 | 0.0208 | 0.0178 |
| FRIDA (FRIDA) | 0.0109 | 0.0272 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0084 | 0.0203 | 0.0149 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0054 | 0.0168 | 0.0069 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0099 | 0.0054 |

#### `substring::precision@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0178 | 0.0059 |
| BGE-M3 (bge-m3) | 0.0059 | 0.0099 | 0.0198 |
| FRIDA (FRIDA) | 0.0000 | 0.0099 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0040 | 0.0139 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0119 | 0.0158 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0178 | 0.0059 |

#### `substring::recall@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0050 | 0.0198 | 0.0198 |
| BGE-M3 (bge-m3) | 0.0248 | 0.0545 | 0.0446 |
| FRIDA (FRIDA) | 0.0396 | 0.0644 | 0.0644 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0594 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0149 | 0.0594 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0050 | 0.0198 | 0.0198 |

#### `substring::recall@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0099 | 0.0198 | 0.0198 |
| BGE-M3 (bge-m3) | 0.0248 | 0.0545 | 0.0446 |
| FRIDA (FRIDA) | 0.0545 | 0.0743 | 0.0743 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0297 | 0.0693 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0198 | 0.0594 | 0.0396 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0099 | 0.0198 | 0.0198 |

#### `substring::recall@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0198 | 0.0099 |
| BGE-M3 (bge-m3) | 0.0099 | 0.0248 | 0.0347 |
| FRIDA (FRIDA) | 0.0000 | 0.0297 | 0.0594 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0297 | 0.0594 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0297 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0198 | 0.0099 |

#### `token_f1::mrr`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0036 | 0.0484 | 0.0185 |
| FRIDA (FRIDA) | 0.0026 | 0.0113 | 0.0032 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0021 | 0.0124 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0022 | 0.0149 | 0.0114 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::ndcg@3`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0495 | 0.0099 |
| FRIDA (FRIDA) | 0.0000 | 0.0099 | 0.0000 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0000 | 0.0099 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0099 | 0.0099 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::precision@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0010 | 0.0158 | 0.0069 |
| FRIDA (FRIDA) | 0.0020 | 0.0030 | 0.0010 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0010 | 0.0030 | 0.0030 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0040 | 0.0020 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::precision@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0020 | 0.0079 | 0.0079 |
| FRIDA (FRIDA) | 0.0015 | 0.0015 | 0.0015 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0015 | 0.0015 | 0.0015 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0015 | 0.0020 | 0.0020 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::precision@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0297 | 0.0079 |
| FRIDA (FRIDA) | 0.0020 | 0.0040 | 0.0020 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0000 | 0.0059 | 0.0059 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0079 | 0.0040 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::recall@10`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0099 | 0.0693 | 0.0396 |
| FRIDA (FRIDA) | 0.0099 | 0.0198 | 0.0099 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0099 | 0.0198 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0297 | 0.0099 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::recall@20`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0297 | 0.0693 | 0.0693 |
| FRIDA (FRIDA) | 0.0198 | 0.0198 | 0.0198 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0198 | 0.0198 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0297 | 0.0297 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

#### `token_f1::recall@5`

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0594 | 0.0297 |
| FRIDA (FRIDA) | 0.0099 | 0.0099 | 0.0099 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0000 | 0.0198 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0297 | 0.0099 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |

### Ограничения (не снимать при публикации)

- Gold v2 без стабильных `chunk_id` — substring/token_f1/semantic по тексту, не BEIR id-hit.
- `semantic::` и `substring::` — разные определения hit; не сравнивать абсолютные уровни между ними.
- Финальный отчёт строится **только** из канонического `run_dir` в протоколе; старые B3 — архив.

<!-- FINAL_RESEARCH_AUTO_END -->





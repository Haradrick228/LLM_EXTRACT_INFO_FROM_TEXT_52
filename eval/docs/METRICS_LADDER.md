# Лестница метрик ретрива / RAG (METRICS_LADDER)

Документ фиксирует **контракт оценки**: что сравниваем, в каком порядке читаем цифры, и как не путать fixture-smoke с полным корпусом.

## Слой 0 — Контекст прогона (обязательная подпись)

В логах смотрите баннер `[eval_config]` из `run_retrieval_eval` и `_meta` в `retrieval.json`:

- `chroma_collection`, **`chroma_collection_count`**, `fetch_k`, `k`, `metric_modes`, `chunking_label`, `gold_sha256`
- **`eval_scenario_type`** — задаётся env `EVAL_SCENARIO_TYPE` (например `full_corpus`, `fixture_smoke`, `external_sanity`); если `(unset)`, явно не сравнивайте с другими прогонами как «тот же сценарий».

**Правило:** не интерпретировать recall@k между прогонами с разным `fetch_k`, разным корпусом или разным `eval_file` как «силу одной модели».

Скан **полного** corpus JSONL для класса **A** может занять время на десятках тысяч строк; не используйте `--max-corpus-lines` для финального отчёта (иначе класс A занижается).

## Слой 1 — Таксономия A–D (диагностика)

CLI: `python -m eval.diagnostics.metrics_ladder_funnel` — читает `trace.jsonl` + gold, опционально corpus JSONL.

|-------|--------|
| **A** | Текст reference (по выбранному `hit_fn`) **не встречается ни в одном** чанке экспортированного корпуса → проблема **gold ↔ чанкинг / нормализация**, не «плохой BGE». |
| **B** | В корпусе есть hit, но **нет** среди top-`fetch_k` до реранка → слабый **первый этап** или маленький `fetch_k`. |
| **C** | Hit есть в пуле `fetch_k`, но **нет** в top-`k_primary` после пайплайна (вкл. реранк) → **ранг / реранк**. |
| **D** | Hit в top-`k_primary` после пайплайна. |
| **A_skip** | Не передан `--corpus-jsonl` или пустые `references` — класс **A** не считался. |

## Слой 2 — Wilson CI и условные метрики

В JSON отчёта `metrics_ladder_funnel`:

- **Wilson 95%** для бинарного «hit@k_post по запросу».
- **`conditional_given_oracle_pool`**: средние recall/MRR **только** там, где gold попал в пул (oracle pre). Отделяет качество ранжирования от «не попал в fetch_k».

## Слой 3 — Реранк

Если в trace есть `stage_02_rerank`, в отчёте появятся средние **ΔMRR**, **Δrecall@k** и доля `match_ambiguous`.

## ID-qrels (gold_v2)

Текущий `eval/datasets/eval_set_gold_v2.jsonl` **не содержит** стабильных `chunk_document_id` из индексируемого корпуса — релевантность задаётся **текстом** `references` / `reference_quote`. Для строгого BEIR-подобного eval нужна пересборка gold с id чанка из того же JSONL, что ушёл в Chroma.

## Негативный контроль

```bash
python -m eval.diagnostics.shuffle_eval_questions_neg_control \
  --in-jsonl eval/datasets/eval_set_gold_v2.jsonl \
  --out-jsonl eval/runs/_neg_control/shuffled_questions_gold_v2.jsonl
```

Дальше `run_retrieval_eval` на `--eval-file` этого файла: метрики на большом корпусе должны **рухнуть** относительно честного gold.

## Опционально: DeepSeek / LLM-слой

Заготовка (без gate в CI): `python -m eval.diagnostics.llm_chunk_relevance_deepseek --dry-run`. Реализация — рубрика + кэш JSONL, см. план `retrieval_metrics_ladder`.

## Пример артефакта (полный корпус, 2026-05)

После прогона `run_retrieval_eval` с `--retrieval-trace-jsonl`:

```bash
python -m eval.diagnostics.metrics_ladder_funnel \
  --trace eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/trace.jsonl \
  --gold-jsonl eval/datasets/eval_set_gold_v2.jsonl \
  --corpus-jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl \
  --out-json eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/metrics_ladder_with_corpusA.json
```

На этом снапшоте (substring, `k_primary=10`): **~59% класс A** — эталонная цитата **не встречается** как подстрока ни в одном экспортированном чанке; **~33% класс B** — в корпусе есть, но не попала в top-`fetch_k`; **~6% D** — hit@10. Условно на пуле (`oracle_pool`, n=8) средний recall@10 post **~0.56** — отдельно про **ранжирование**, когда gold вообще достижим в пуле.

## Кривая fetch_k

См. артефакты sweep, например `eval/runs/20260513_015456_f2_hybrid_rrf_sweep/` (матрица `fetch_k` × режимы).

## Калибровка semantic

Порог `RETRIEVAL_SEMANTIC_THRESHOLD` / `--semantic-threshold` держите фиксированным между A/B; для нового корпуса — мини-срез 20–30 пар «чанк — релевантен ли» и подстройка порога (ручной шаг, не автомат в CI).

**Автоматический зонд (топ-1 чанк × primary reference):**

```bash
python -m eval.diagnostics.semantic_threshold_probe \\
  --trace eval/runs/<id>/trace.jsonl \\
  --gold-jsonl eval/datasets/eval_set_gold_v2.jsonl \\
  --max-rows 30 \\
  --out-json eval/runs/<id>/semantic_probe.json
```

Смотрите `percentiles_substring_hit` vs `percentiles_substring_miss` и `scorer_backend` (если `substring-fallback`, установите `sentence-transformers`).

## Answer-eval в каталоге retrieval-прогона

Выровнять Chroma / `fetch_k` / `eval_file` с `retrieval.json` → `_meta`:

```bash
python -m eval.diagnostics.run_answer_eval_in_run_dir \\
  --run-dir eval/runs/<id> \\
  --max-rows 25
```

Пишет `answer_eval_manifest.json` и `answer_eval_results.jsonl` (см. `eval/retrieval/run_answer_quality_eval.py`; нужен рабочий LLM/RagService).

## LLM релевантность чанков (DeepSeek)

Кэш-аппенд JSONL, не для CI без ключа:

```bash
python -m eval.diagnostics.llm_chunk_relevance_deepseek \\
  --trace eval/runs/<id>/trace.jsonl \\
  --out-jsonl eval/runs/<id>/llm_chunk_relevance.jsonl \\
  --max-queries 15 --top-chunks 2
```

`--dry-run` — без сети, пишет помеченные строки для проверки пайплайна.

## Слой answer-eval (прямой CLI и обёртка по run_dir)

- Прямой запуск: `python -m eval.retrieval.run_answer_quality_eval` — env `CHROMA_*`, `ANSWER_EVAL_FETCH_K`, `ANSWER_EVAL_FILE` (см. заголовок скрипта).
- Выровнять с retrieval-прогоном: см. выше **Answer-eval в каталоге retrieval-прогона** (`run_answer_eval_in_run_dir`).

Не смешивайте answer-sim с recall@k без отдельной колонки в отчёте.

## Ссылки на код

- Агрегаты: `core/rag/metrics/ranking_aggregates.py`
- Трейс + реранк: `eval/retrieval/run_retrieval_eval.py`
- Воронка: `eval/diagnostics/metrics_ladder_funnel.py`
- Semantic зонд: `eval/diagnostics/semantic_threshold_probe.py`
- Answer-eval в run_dir: `eval/diagnostics/run_answer_eval_in_run_dir.py`
- LLM relevance: `eval/diagnostics/llm_chunk_relevance_deepseek.py`

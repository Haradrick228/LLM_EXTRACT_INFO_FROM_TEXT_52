# Фаза B3 — реранкеры на полном корпусе (после B1 full + B2 по желанию)

Документ‑памятка: **выполнять только после** того, как все коллекции из `eval/phase_b/embedding_bases_full_corpus.yaml` заполнены до одного и того же `N` (сейчас **пять** dense-моделей без Giga-Embeddings-instruct; ожидается **61 543** строки экспорта из `default`).

## Цель

- **Пул поиска:** полный корпус в Chroma (`corpus_phase_b_mpnet`, … из `eval/phase_b/embedding_bases_full_corpus.yaml`).
- **Оценка качества:** тот же золотой набор вопросов **`eval/eval_set_gold_v2.jsonl`** (**101** строка) — метрики retrieval/rerank по substring/recall/MRR/nDCG как в `run_retrieval_eval`.
- **Матрица:** все профили `rag_profiles.yaml` с `matrix: true` × **каждая** dense‑коллекция из реестра (по числу строк в YAML).

## Предпосылки (чеклист)

1. `python -m eval.phase_b.verify_corpus_pipeline index-align --expected-rows 61543` — exit 0 (подставь актуальный `N` из `export_chroma_run` manifest).
2. `CHROMA_EMBED_MODEL` в окружении не должен путать сценарии: оркестратор B3 передаёт `--embed-model` и `--chroma-collection` **явно** для каждой пары.
3. VRAM: каждый сценарий грузит свой SentenceTransformer; между сценариями процесс один — при OOM уменьшать `ST_EMBED_BATCH_SIZE` / профили реранкера.

## Команда B3

```text
python -m eval.phase_b.run_full_corpus_rerank_matrix --expected-rows 61543
```

Опции: `--dense-registry`, `--profiles`, `--eval-file`, `--fetch-k`, `--k`, `--retrieval-max-rows`, `--skip-answer`.

Replay (после мёрджа в реестр):

```text
python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix
```

## Артефакты

`eval/runs/<timestamp>_b3_full_corpus_rerank/` — `summary.json`, подкаталоги `full_mpnet/`, `full_bge_m3/`, … с `retrieval_<profile>.json` и логами stdout/stderr.

## Связь с B2 (фикстура)

Слоты `phase_b_dense_slot_*` в `experiments_registry.yaml` остаются для **gold_v2_fixture** / `phase_b_gold_v2_*` (~101 чанк). B3 — **отдельный** сценарий на `corpus_phase_b_*`; путать нельзя.

## Параметры `fetch_k`

На полном корпусе дефолт `40` может быть мало относительно 61k; для сравнения реранкеров имеет смысл заранее зафиксировать `RETRIEVER_FETCH_K` / `--fetch-k` и записать в `summary` через баннер (уже есть `fetch_k` в summary).

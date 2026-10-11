# Фаза B — шесть Chroma-баз на одной фикстуре gold v2

Все шесть баз — **SentenceTransformers / HuggingFace** (один и тот же путь в Chroma query: `eval/chroma_utils.get_collection` + `SentenceTransformerEmbeddingFunction`).

**Чанки vs эмбеддинги vs ретривер:** на фикстуре фазы B текст чанков **один** (`eval/fixtures/chunks_sample.jsonl`); для каждой HF-модели заново считаются **только вектора** по этим же строкам → отдельная коллекция Chroma на модель. Это ровно сценарий «из чанков — новые эмбеддинги разными моделями». Ретривер — следующий слой (поиск по уже посчитанным векторам); про него говорят, когда сравнивают качество через eval, а не когда строят индекс.

### Фикстура (~101 чанка) vs полный корпус (~десятки тысяч)

- **Gold-фикстура** (`index_gold_fixture_bases` + `chunks_sample.jsonl`) — ровно **101** строка: общий эталон для eval по gold v2, не размер продакшен-корпуса.
- **Полный корпус** — отдельный поток: экспорт **всех** чанков из эталонной коллекции (например `default_clean`) в JSONL, затем шесть коллекций с префиксом `corpus_phase_b_*` и теми же шестью моделями (`index_jsonl_bases`).

```text
python -m eval.export_chroma_run --collection default_clean --run-id corpus_export_<ts>
python -m eval.phase_b.index_jsonl_bases --jsonl DATA/runs/corpus_export_<ts>/chunks_from_chroma.jsonl --all-st --rebuild
```

Реестр полного корпуса: `eval/phase_b/embedding_bases_full_corpus.yaml`. Пакетный `add` в Chroma: `PHASE_B_CHROMA_ADD_BATCH` (по умолчанию 256). Смок: `--max-rows N`.

## Где что лежит

| Что | Путь |
|-----|------|
| Реестр 6 баз | `eval/phase_b/embedding_bases.yaml` |
| Полный корпус: реестр 6 коллекций | `eval/phase_b/embedding_bases_full_corpus.yaml` |
| Индексация фикстуры + манифесты | `python -m eval.phase_b.index_gold_fixture_bases --help` |
| Индексация полного JSONL + манифесты | `python -m eval.phase_b.index_jsonl_bases --help` |
| Фикстура чанков | `eval/fixtures/chunks_sample.jsonl`, загрузчик `eval/phase_b/fixture_chunks.py` |
| Sync только mpnet | `scripts/sync_gold_v2_fixture_collection.py` → `gold_v2_fixture` |
| Replay-эксперименты | `eval/experiments_registry.yaml` |

Артефакты индексации: `DATA/runs/phase_b_index/<timestamp>/run.log`, `manifest.json`, `bases/<base_id>.json` (`retrieval_env_hint`: `CHROMA_COLLECTION` + `CHROMA_EMBED_MODEL`).

## Шесть моделей (обоснование)

| # | Коллекция | HF-модель | Зачем в наборе |
|---|-------------|-----------|----------------|
| 1 | `gold_v2_fixture` | `paraphrase-multilingual-mpnet-base-v2` | Baseline проекта. |
| 2 | `phase_b_gold_v2_bge_m3` | `BAAI/bge-m3` | Запрошено; уже сильный на вашем корпусе (README). |
| 3 | `phase_b_gold_v2_frida` | `ai-forever/FRIDA` | Запрошено; ru-задачи MTEB / MIRACL ru, RiaNews, RuBQ и др. (карточка на HF). |
| 4 | `phase_b_gold_v2_e5_large_instruct` | `intfloat/multilingual-e5-large-instruct` | Сильный multilingual по **MMTEB** (публичные сводки 2025) и общей практике RAG. |
| 5 | `phase_b_gold_v2_jina_embeddings_v3` | `jinaai/jina-embeddings-v3` | Актуальный multilingual dense на **MTEB-ориентированных** сравнениях. |
| 6 | `phase_b_gold_v2_giga_embeddings_instruct` | `ai-sage/Giga-Embeddings-instruct` | Запрошено **вместо GigaChat API**; локальный ST, `trust_remote_code`; в реестре **последняя** по порядку (`heavy_index_last`) — тяжёлая/кастомная, индексировать после остальных. |

Дополнительно по русскому: в литературе/репозитории **MTEB** обсуждаются **ruMTEB** и русскоязычные задачи (см. PR «Add Russian tasks (RU-MTEB)» к embeddings-benchmark/mteb, NAACL 2025 «Russian-focused embedders' exploration»). FRIDA и multilingual лидеры (E5, Jina, BGE-M3) — стандартный набор для сравнения без дублирования API.

### Giga-Embeddings-instruct и GPU

Это **не** облачный GigaChat. Если PyTorch ругается на архитектуру GPU (например Blackwell / sm_120), уменьшайте **`ST_EMBED_BATCH_SIZE`**, задавайте **`ST_EMBED_DEVICE`**, при необходимости обновляйте PyTorch/CUDA или кэшируйте веса в `HF_HOME`; кастомный код модели требует **`trust_remote_code: true`** (уже в YAML).

## B1 — индексация

Одной командой все шесть (порядок из YAML, Giga — последняя):

`python -m eval.phase_b.index_gold_fixture_bases --all-st`

или `python -m eval.replay_experiment run phase_b_index_fixture_bases_st`

Пересборка одной базы: `--base-id <base_id> --rebuild`. **`gold_v2_fixture`** без `--force-include-default-fixture` не удаляем.

## B2 — матрица реранкеров

Слоты **`phase_b_dense_slot_1` … `_6`** в `eval/experiments_registry.yaml` (дефолты совпадают с таблицей выше). Это **фикстура** (~101 чанк в коллекциях `gold_v2_fixture` / `phase_b_gold_v2_*`), если не переопределять env.

## B3 — реранкеры на полном корпусе (`corpus_phase_b_*`)

После того как `index_jsonl_bases` заполнил все шесть коллекций полного корпуса до одного `N`, матрица реранкеров по **тем же 101 вопросам** `eval_set_gold_v2.jsonl`, но с поиском по **61k+** чанкам: план **`eval/docs/PHASE_B3_FULL_CORPUS_RERANK_PLAN.md`**, оркестратор **`python -m eval.phase_b.run_full_corpus_rerank_matrix`**, replay **`phase_b3_full_corpus_rerank_matrix`**.

## Прочее

| Назначение | Путь |
|------------|------|
| Профили реранкера | `eval/rag_profiles.yaml` |
| Матрица | `eval/run_profile_metrics.py` |

**B1/B2** — не гонять без запаса VRAM/диска.

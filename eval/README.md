# Eval артефакты

## Датасеты
- **`eval_set_gold_v2.jsonl`** + **`eval_set_gold_v2.meta.json`** — golden v2. Схема строки:
  - `question` — вопрос по тексту чанка;
  - **`reference_quote`** — прямая дословная цитата из чанка (обязательно для корпуса из Chroma);
  - `references` — список цитат для метрик retrieval (обычно `[reference_quote]`);
  - `answer_ref` — ожидаемый ответ (для **chroma_curated** — своя формулировка; для **synthetic** — весь текст чанка);
  - `chunk_id`, `source`.
- **Реальный gold без локальной нейросети (основной путь):** перенос из уже размеченного корпусного набора  
  `python -m eval.materialize_gold_v2_from_answer_ref --n 100 --yes`  
  Берётся **`eval/eval_set_with_answer_ref.jsonl`** (вопросы и цитаты из ваших материалов). Для каждой строки строится чанк: `answer_ref` + недостающие `references`, чтобы все цитаты были **подстроками** `document` (нужно для `run_retrieval_eval`). Поле **`reference_quote`** = первая цитата. В `meta`: **`corpus_type: corpus_migrated`**, LLM не используется. Проверка: `python -m eval.validate_gold_v2`.
- **Альтернатива с Chroma + Ollama** (если понадобится автодобор новых чанков): `python -m eval.build_gold_v2_from_chroma --collection default --n 50 --yes` → `corpus_type: chroma_curated`.
- **Синтетика (только автотесты / CI):** `python eval/generate_gold_v2_corpus.py --n 100` — `corpus_type: synthetic`.
- `fixtures/chunks_sample.jsonl` — чанки, **ровно те**, что соответствуют строкам gold v2 (после сборки из Chroma или синтетики).
- Первый профиль матрицы реранкеров: **`baseline_mpnet_rerank`**. Синк в отдельную коллекцию Chroma (если volume доступен на запись): `python scripts/sync_gold_v2_fixture_collection.py` → **`gold_v2_fixture`**.
- `eval_set.jsonl` — 101 вопрос без `answer_ref`.
- `eval_set_with_answer_ref.jsonl` — 101 вопрос с `answer_ref` (основной eval). Поле `references` — список релевантных чанков/файлов, на них считаются метрики поиска.
- `eval_set_with_answer_ref_sample20.jsonl` — 20 вопросов для быстрых прогонов/локальных моделей.
- `corpus_sample.jsonl` — 500 документов из Chroma (выборка для ручной проверки).

## Запуск с хоста Windows (Docker уже поднят)

В [`.env`](../.env) для docker-compose часто задано `CHROMA_HOST=chroma` и порт `8000` — это **имя сервиса внутри сети compose**. Если `uvicorn`/`RagService` запускаете **на машине**, а не в контейнере `app`, выставьте для Chroma:

- `CHROMA_HOST=127.0.0.1`
- `CHROMA_PORT=18000` (проброс из `infra/docker-compose.yml`: `18000:8000`)

Если в `.env` для compose остаётся `CHROMA_HOST=chroma`, для **eval с хоста** задайте: `RAG_EVAL_CHROMA_HOST=127.0.0.1`, `RAG_EVAL_CHROMA_PORT=18000` (так подключаются `run_retrieval_eval`, `run_answer_eval`, `run_profile_metrics`).

Ollama в Docker у вас на `127.0.0.1:11434`; для LLM: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`.

Быстрая проверка без PyTorch/HF: `python scripts/verify_local_stack.py`

Смоук фазы A (профиль `baseline_mpnet_rerank`, `eval_set_gold_v2`, без answer-eval): `python scripts/smoke_phase_a.py`

**Полный смоук (без фазы B):** `python scripts/full_smoke.py` — проверка стека, затем retrieval на `eval_set_with_answer_ref_sample20.jsonl` (10 вопросов), артефакты в `eval/runs/<ts>_full_smoke/`. С цепочкой LLM (2 вопроса, Ollama): `python scripts/full_smoke.py --with-llm`. Параметры: `--retrieval-rows`, `--answer-rows`.

Интеграционные pytest (нужен поднятый Chroma): `CHROMA_INTEGRATION=1 python -m pytest tests/test_integration_chroma.py -v`

Golden v2 + первый реранкер матрицы (`baseline_mpnet_rerank`): `python scripts/run_gold_v2_baseline_retrieval.py` (результат по умолчанию `eval/runs/gold_v2_baseline_mpnet_rerank.json`). Если Chroma пишет в volume без readonly, для строгого совпадения с фикстурой: `python scripts/sync_gold_v2_fixture_collection.py` затем `CHROMA_COLLECTION=gold_v2_fixture python scripts/run_gold_v2_baseline_retrieval.py`.

Эмбеддеры и реранкеры грузятся на **GPU** при `RERANK_DEVICE=cuda` и CUDA-доступном PyTorch; при ошибках загрузки весов с `huggingface.co` проверьте сеть или локальный кэш `HF_HOME` / `TRANSFORMERS_OFFLINE=1`.

## Воспроизводимые эксперименты (реестр)

Именованные прогоны и их окружение лежат в **[`experiments_registry.yaml`](experiments_registry.yaml)**. Меняете YAML (или подставляете `-e KEY=VAL`) — повторяете тот же смысловой эксперимент без переписывания команд.

- Список: `python -m eval.replay_experiment list`
- Запуск: `python -m eval.replay_experiment run <experiment_id>` (подкоманды `list` и `run`)
- Проверка команды и env без исполнения: `... run <id> --dry-run`
- Разовые переопределения: `... run retrieval_matrix_gold_v2_fixture -e CHROMA_COLLECTION=my_collection`
- Другой файл реестра: `EXPERIMENTS_REGISTRY=path/to.yaml` или `--registry path/to.yaml`

Реализация: [`replay_experiment.py`](replay_experiment.py) (слияние `defaults` по `inherit`, подстановки `${VAR:-default}`). Профили реранкеров/LLM по-прежнему в [`rag_profiles.yaml`](rag_profiles.yaml); матрица фазы A — эксперимент `retrieval_matrix_gold_v2_fixture` (`eval.run_profile_metrics`). Фаза B (три dense-коллекции) — слоты `phase_b_dense_slot_*` и [`PHASE_B_SCOPE.md`](PHASE_B_SCOPE.md).

## Профили RAG / реранкер (фаза A)
- Файл [`rag_profiles.yaml`](rag_profiles.yaml) — именованные наборы переменных окружения (`RERANK_MODEL`, `RERANK_MODE`, опционально `LLM_PROVIDER=ollama` и т.д.).
- Модуль [`../common/rag_profile_env.py`](../common/rag_profile_env.py) — `apply_profile(name)`, `list_profile_names(matrix_only=True)`, `apply_profile_by_rag_profile_env()` (читает `RAG_PROFILE`).
- Запуск ML-сервиса с профилем: из корня репозитория  
  `python -m eval.run_with_profile --profile rerank_jina_v3 -- uvicorn ml_service.main:app --host 0.0.0.0 --port 8000`  
  либо выставить `RAG_PROFILE=rerank_jina_v3` в окружении перед `uvicorn` (профиль применяется при `create_app()`).
- Локальные ответы RAG / answer-eval: `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434`, модель по умолчанию `qwen2.5:7b-instruct` (`LLM_PROVIDER_MODEL` / `OLLAMA_MODEL`). Профиль `ollama_answer_eval` в YAML задаёт это одним именем.

## Скрипты
- `run_retrieval_eval.py` — метрики поиска: recall@k, precision@k (k=5/10 по умолчанию), MRR, nDCG@3.
  - nDCG: бинарная релевантность по `references`, DCG с делителем `log2(rank+1)`, ранжирование — по скору эмбеддера/реранкера.
  - precision/recall считают совпадение строки чанка с любым `references` (регистр игнорируется, подстрока допустима).
- `run_answer_eval.py` — оценка ответов по `answer_ref` и `references`.
- `run_profile_metrics.py` — поочерёдно применяет профили с `matrix: true` из `rag_profiles.yaml`, запускает `run_retrieval_eval` и (опционально) `run_answer_eval`, пишет в `eval/runs/<timestamp>_rerank_matrix/` (старые прогоны не трогает).
- `export_chroma_run.py` — экспорт коллекции Chroma в `DATA/runs/<run_id>/chunks_from_chroma.jsonl` + `manifest.json` (при необходимости можно собрать выборку вроде `corpus_sample.jsonl` из этого экспорта).
- `reindex_clean.py` — реиндексация `default_clean` с альтернативным эмбеддером.

## Результаты
- `rerank_results.md` — таблица сравнений реранкеров/конфигураций.
- `answer_eval_results.jsonl` — прогон 101 вопрос (DeepSeek API). Снимок предсказаний: `archive/eval_research/snapshots/answer_eval_predictions.jsonl`.
- `answer_eval_results_ollama_sample20.jsonl` — прогон 20 вопросов на `qwen2.5:7b-instruct` через Ollama.
- `ANSWER_EVAL_SUMMARY.md` — краткая сводка ответных метрик и ссылки на исходные файлы.
- Исследовательские ноутбуки и сырые снимки метрик: каталог `archive/eval_research/` (см. `archive/README.md`). Обзорный `eval_report.ipynb` ищет `eval/` от корня репозитория.

## Удалено/не используется
- `answer_eval_predictions_sample.jsonl` — битый дубликат, больше не используем.

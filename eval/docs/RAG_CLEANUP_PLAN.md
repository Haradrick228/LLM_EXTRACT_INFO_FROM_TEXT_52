# План очистки RAG-части репозитория (расширенная инвентаризация)

Предыдущая версия плана была слишком общей. Ниже — **обзор по фактам из кода и `grep` по импортам** (без удалений; только классификация и рекомендации).

**Границы:** фокус на «обычном RAG» (`core/rag/`, `eval/`, `scripts/`, `common/`, точки входа `ml_service/`, боты, скрейпер). **`core/graphrag/`** не переписываем в рамках очистки, но **учитываем**, что `rag_service.py` опционально тянет `GraphRetriever` и `build_llm_client` — ломать эти импорты нельзя.

**Принципы:** как и раньше — сначала согласованный список, потом перенос в `archive/` или политика git; артефакты `eval/runs/` и золотые наборы не выбрасываем без явного решения.

---

## 1. «Позвоночник» продакшена и интеграций

```text
ml_service.api.create_app()
  → common.rag_profile_env.apply_profile_by_rag_profile_env()   [опционально]
  → core.rag.rag_service.RagService()                             [основной API-сервис]

bots/telegram, bots/rest_api, scraper/website_scraper
  → core.rag.rag_service_wrapper.RagServiceWrapper
       → RagService(...) + QAService + DeepSeekLLM (логирование / черновик)
```

**Вывод:** `rag_service.py` и `rerank_registry.py` — ядро; `rag_service_wrapper.py` — контракт для ботов и скрейпера. Версия с `ValidationService` (редактор ответа): **`archive/core_rag/rag_service_wrapper_with_validation.py`**.

---

## 2. `core/rag/` — построчно

| Файл | Роль | Кто тянет | Заметки / риск снятия |
|------|------|-----------|------------------------|
| `rag_service.py` | Chroma, эмбеддинг, реранк, ответ LLM, опционально graph | `ml_service`, wrapper, eval через тот же класс | Крупный; любая «чистка» только с тестами и смоуком. |
| `rerank_registry.py` | Фабрика CrossEncoder / адаптеры | `rag_service` | Часть матрицы профилей. |
| `query_cache.py` | Кэш запросов | `rag_service` | Не выкидывать без проверки env `LLM_CACHE_ENABLED`. |
| `deepseek_llm.py` | Клиент для обёртки / legacy | `rag_service_wrapper` (+ старый wrapper) | Зависит от ключей; для API-сервиса может быть не нужен, но боты завязаны. |
| `rag_service_wrapper.py` | Обёртка: RAG + QA log + нормализация списков | `bots/*`, `scraper` | **Прод-путь.** |
| *(перенесено)* `archive/core_rag/rag_service_wrapper_with_validation.py` | То же + `ValidationService.rewrite` | Не импортируется продом | Референс post-edit LLM. |

---

## 3. `eval/` — не только «три скрипта метрик»

### 3.1 Скрипты с явной ценностью (часто недооцениваются)

| Модуль | Зачем нужен | Импорты из кода |
|--------|-------------|-----------------|
| `text_pipeline.py` | `clean_text` / `chunk_text` — **контракт** для тестов и согласованности с чанкингом | `tests/test_text_pipeline.py` |
| `chroma_utils.py` | Подключение к Chroma для eval / интеграций | `run_retrieval_eval`, `run_answer_eval`, тесты |
| `export_chroma_run.py` | Экспорт коллекции в `DATA/runs/<run_id>/` (lineage для gold / отладки) | Документирован в `eval/README.md`; прямых импортов мало — **CLI-инструмент** |
| `reindex_clean.py` | Дедуп + перенос в `default_clean` / другой embedder | Документирован в README; **операционный**, не мёртвый |
| `materialize_gold_v2_from_answer_ref.py` | Основной путь gold без LLM из `eval_set_with_answer_ref.jsonl` | CLI |
| `build_gold_v2_from_chroma.py` | Альтернативный путь gold из Chroma + Ollama | CLI, перекрёстные ссылки в `generate_gold_v2_corpus.py` |
| `validate_gold_v2.py` | Проверка согласованности gold ↔ fixtures | CLI, pytest |
| `run_with_profile.py` | Запуск произвольной команды с профилем из YAML | Док, ручной запуск uvicorn |
| `profile_env.py` | Реэкспорт из `common.rag_profile_env` | Обратная совместимость импортов |

### 3.2 Данные и отчёты (исследование, не «мусор»)

| Артефакт | Смысл |
|----------|--------|
| `rag_optimization_data.py` | Словари метрик для ноутбука `archive/eval_research/notebooks/rag_optimization_report.ipynb` |
| `embedder_comparison.json`, `frida_*`, … | Снимки в `archive/eval_research/snapshots/` (эксперименты по чанкингу/эмбеддерам) |
| `rerank_results.md`, `ANSWER_EVAL_SUMMARY.md`, `answer_eval_plan.md`, `PHASE_B_SCOPE.md` | Человеческие выводы и границы фаз |
| `notebooks/*.ipynb` | Визуализация; связаны с данными выше |

**Рекомендация:** не сливать в одну кучу с «логами прогона»; при архивации — **паковать по теме** (embedder vs rerank vs answer).

### 3.3 `eval/runs/**`

Несколько каталогов `*_rerank_matrix` — это **не дубликаты бессмысленные**, а отдельные запуски (env snapshot, stdout/stderr). Сжатие — только после сравнения `summary.json` / даты; иначе теряется трассировка среды.

### 3.4 Дрейф документации (зафиксировать в фазе «док», не удаление)

В `eval/README.md` в блоке про `eval_set_gold_v2.jsonl` до сих пор перечислены поля вроде `chunk_id`/`source` в описании строки; фактическая схема после materializer может отличаться. **Очистка = привести README к одному источнику правды**, а не удалять jsonl.

---

## 4. `common/`

| Файл | Роль |
|------|------|
| `rag_profile_env.py` | `apply_profile`, `RAG_PROFILE`, используется **`ml_service` при старте** |
| `rag_constants.py` | Сепараторы/defaults; комментарии указывают на `archive/legacy_indexing/migrate_to_chroma.py` |

Удаление или переименование без рефакторинга всех вызовов = поломка сервиса и eval.

---

## 5. Корень репозитория

| Путь | Классификация |
|------|----------------|
| `archive/legacy_indexing/migrate_to_chroma.py` | **Legacy LangChain → Chroma** (порты по умолчанию 8000, не 18000); не импортируется как библиотека; см. `common/rag_constants.py`. |
| `ml_service.db` | Локальная SQLite для логов/истории — **не коммитить** (`.gitignore`), не путать с «мусором». |
| `rag (1).zip` | Неизвестное содержимое — только после просмотра: архив или ignore. |

---

## 6. `scripts/` (все `.py` в репо под `scripts/`)

| Скрипт | Назначение |
|--------|------------|
| `verify_local_stack.py` | Быстрая проверка Chroma/стека без тяжёлого PyTorch |
| `smoke_phase_a.py`, `full_smoke.py` | Регрессия профилей + retrieval (и опционально LLM) |
| `sync_gold_v2_fixture_collection.py` | Коллекция `gold_v2_fixture` для строгого eval |
| `run_gold_v2_baseline_retrieval.py` | Бейзлайн-прогон gold |
| `download_embedders.py` | Предзагрузка весов |
| `run_retrieval_eval_queued.ps1`, `up_infra.ps1` | Очередь GPU / Docker |

Ни один из них не «лишний» без отдельного аудита сценариев CI/локалки.

---

## 7. `archive/` и дубли с корнем

В `archive/` лежат десятки вариантов `reindex_*`, `run_*_local*.py` — это **журнал попыток GPU/Chroma**, не дубликаты текущего `eval/run_retrieval_eval.py`. Имеет смысл **оставить** как историю; при желании — один `archive/README.md` со списком «что смотреть первым» (не делалось в этом PR — по запросу).

---

## 8. Кандидаты на действия (после вашего апрува)

**Низкий риск, высокая ясность**

1. Синхронизировать `eval/README.md` с актуальной схемой gold v2 и с `--progress-log` / `run_retrieval_eval_queued.ps1`.
2. Дописать в `eval/RAG_CLEANUP_PLAN.md` или короткий указатель: «источник правды по gold — `eval_set_with_answer_ref.jsonl` → materialize → validate».

**Перенос без удаления**

3. ~~Перенос старого wrapper~~ сделан: `archive/core_rag/rag_service_wrapper_with_validation.py`, см. docstring в `core/rag/rag_service_wrapper.py`.

**Политика артефактов**

4. `eval/runs/`: не удалять автоматически; опционально zip по кварталам в `DATA/` **копией**, если репозиторий раздувается.

**Исследовательский контур (уже в `archive/eval_research/`)**

- `rag_optimization_data.py`, снимки `frida_*` / `retrieval_bge-m3_*`, ноутбуки — в архиве; в активном `eval/` остаются `reindex_clean.py`, `export_chroma_run.py`, датасеты и скрипты метрик.

---

## 9. Критерии «очистка не сломала RAG»

- `pytest tests/` (по необходимости без `CHROMA_INTEGRATION`).
- `python scripts/verify_local_stack.py` и/или `python scripts/smoke_phase_a.py`.
- Ручная проверка: `python -m eval.validate_gold_v2` после правок датасетов.

---

## 10. Вне скоупа

- Логика GraphRAG и схемы графовых индексов.
- Массовое удаление `eval/runs/*` без архивной копии.

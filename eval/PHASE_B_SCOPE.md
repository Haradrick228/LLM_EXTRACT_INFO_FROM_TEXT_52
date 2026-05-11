# Фаза B (отложено — не выполнять без явного апрува)

Этот файл фиксирует объём будущей работы; код индексации под несколько dense здесь **не** запускается автоматически.

## B0 — выбор dense под русский

Совместно выбрать три актуальных эмбеддера (HF / локальные веса) с обоснованием под русскоязычный корпус и доступное железо.

## B1 — данные и Chroma

- Повторный путь: сырой DATA → clean → chunk (при необходимости новые параметры чанка).
- Три **новые** именованные коллекции Chroma (без удаления существующих).
- Manifest и артефакты только в новых `DATA/runs/<run_id>/`.

## B2 — матрица реранкеров

Для **каждой** из трёх коллекций выполнить те же профили реранкеров, что в фазе A (`rag_profiles.yaml`, `matrix: true`), складывая результаты в **новые** `eval/runs/*`, не перезаписывая прогоны фазы A.

---

## Карта файлов (чтобы следующая сессия не искала)

| Назначение | Путь |
|------------|------|
| Именованные прогоны + env + `inherit` | `eval/experiments_registry.yaml` |
| CLI: `list` / `run <id>` / `--dry-run` / `-e` | `eval/replay_experiment.py` |
| Профили реранкера/LLM, матрица `matrix: true` | `eval/rag_profiles.yaml` |
| Оркестратор матрицы (пишет `eval/runs/<ts>_.../`) | `eval/run_profile_metrics.py` |
| Retrieval-метрики | `eval/run_retrieval_eval.py` |
| Gold v2 + валидация | `eval/eval_set_gold_v2.jsonl`, `eval/validate_gold_v2.py` |
| Фикстура в Chroma | `scripts/sync_gold_v2_fixture_collection.py` → коллекция `gold_v2_fixture` |
| Экспорт чанков из Chroma + manifest | `eval/export_chroma_run.py` → `DATA/runs/<run_id>/` |
| Личные заметки исследователя (не в git) | `docs/AGENT_RAG_STACK.md` (см. `.gitignore`) |
| Порты Chroma с хоста Windows | `eval/README.md` — блок «Запуск с хоста» |

## Что дальше (порядок, после явного «делаем B»)

1. **B0 — апрув:** зафиксировать три HF-id dense-эмбеддера под русский и VRAM; записать их в этот файл или в комментарий к реестру.
2. **B1 — индексы:** для каждой пары (эмбеддер, при необходимости чанкинг) — отдельная коллекция Chroma с **уникальным именем**; не трогать `gold_v2_fixture` и рабочие коллекции фазы A. Манифесты — только новые `DATA/runs/<run_id>/`.
3. **B2 — три прогона матрицы:** для коллекции 1/2/3:
   - либо `python -m eval.replay_experiment run phase_b_dense_slot_1_rerank_matrix` (и слоты 2/3), предварительно выставив `CHROMA_COLLECTION_B1` / `CHROMA_EMBED_MODEL_B1` и т.д.;
   - либо три раза `run retrieval_matrix_gold_v2_fixture -e CHROMA_COLLECTION=... -e CHROMA_EMBED_MODEL=...`.
4. **Сводка:** сравнить `eval/runs/*/summary.json` (и при необходимости обновить таблицу в корневом `README.md` / `analysis/reports/eda_dashboard.html`).

**Не стартовать B1/B2 без пункта B0** — иначе три коллекции будут произвольными и сравнение не воспроизводимо.

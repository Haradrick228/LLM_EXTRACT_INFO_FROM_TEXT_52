# Retrieval / RAG — диагноз низких метрик (2026-05)

Документ обновлён по плану «честные доказательства»: **gold на full corpus** с trace, worst-queries, **B3** (частичная матрица), **внешний SQuAD** в изолированной коллекции, правки HTML-отчёта (`token_f1` колонки + секция B3), smoke **F7** после baseline.

---

## 1. Краткий вывод

| Наблюдение | Интерпретация |
|------------|----------------|
| **Dense на full corpus:** `corpus_phase_b_bge_m3`, 101 вопрос, `fetch_k=100`, без реранка → substring **R@10 ≈ 0.0446**, token_f1 **R@10 ≈ 0.0396** (`eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/`). | Это честная база против фикстуры; цифры согласуются с F2/F4 для того же embedder. |
| **Fixture (ошибочно ранее выдано за «доказательство полного корпуса»):** `phase_b_gold_v2_bge_m3`, `--max-rows 25` = только **первые 25** строк gold, индекс **~101 чанк**, не 61k. | Высокий recall там **не** отвечает на вопрос «почему на 60k плохо»; пригоден лишь как sanity кода/Chroma. |
| **Внешний англ. SQuAD:** 80 passage, коллекция `sanity_external_squad_mpnet`, mpnet dense → substring **R@10 = 0.825** (`eval/runs/20260513_sanity_external_squad_mpnet/`). | Пайплайн даёт высокий recall на чистом QA; основной русский корпус + substring остаются **dominant** факторами. |
| **B3 (subset):** `full_bge_m3`, `fetch_k=40`, три matrix-профиля — MiniLM baseline **ломает** substring vs Jina/BGE-v2 на этом срезе (`eval/runs/20260513_210857_b3_full_corpus_rerank/summary.json`). | Реранкер и профиль критичны; см. таблицу в [RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md](RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md). |
| **F7 smoke (12 вопросов):** `eval/runs/20260513_210704_f7_post_diagnosis_smoke` — на малой выборке substring может быть 0 при ненулевом token_f1; не интерпретировать без полного прогона. | Нужен полный F7 или анализ trace на 101 строке. |

---

## 2. Зафиксированный снимок (инварианты)

| Ключ | Значение |
|------|-----------|
| Gold file | `eval/datasets/eval_set_gold_v2.jsonl` |
| `gold_sha256` | `d2d35200b634b92933989c8457f8bc6fe670b9edf7a4ab688ab8e3ae7ae50c96` |
| Строк в gold | 101 |
| BM25 sidecar | `DATA/indexes/bm25/corpus_full_20260512` (61543) |
| HTML-агрегатор | `analysis/reports/retrieval_recovery_2026_05.html` (обновлён: F2–F4, F7, **B3**, колонки **token_f1::recall@10/5**) |
| Gold@full trace run | `eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/` (`trace.jsonl`, `retrieval.json`, `manifest.json`) |
| B3 consolidated | `eval/runs/20260513_210857_b3_full_corpus_rerank/` |
| Индексация: исправлен вызов `_index_streaming(..., resume=False)` в [eval/indexing/index_corpus_from_jsonl.py](indexing/index_corpus_from_jsonl.py) (раньше падало с missing `resume`). | Нужно для любых новых JSONL-индексов, включая sanity. |

`RETRIEVAL_EVAL_SHUFFLE=0`: при записи `--retrieval-trace-jsonl` shuffle отключается в [eval/retrieval/run_retrieval_eval.py](retrieval/run_retrieval_eval.py) (стабильный порядок строк для worst-queries).

---

## 3. Реестр логов и артефактов

| Артефакт | Путь |
|-----------|------|
| Full corpus dense + trace | `eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/trace.jsonl` |
| Worst-queries (markdown) | [RETRIEVAL_WORST_QUERIES_2026_05.md](RETRIEVAL_WORST_QUERIES_2026_05.md) (генерация: `python -m eval.diagnostics.summarize_worst_retrieval_traces ...`) |
| B3 summary | `eval/runs/20260513_210857_b3_full_corpus_rerank/summary.json` |
| SQuAD corpus + gold | `DATA/runs/sanity_external_squad_smoke/` |
| SQuAD eval JSON | `eval/runs/20260513_sanity_external_squad_mpnet/retrieval.json` |
| F7 smoke | `eval/runs/20260513_210704_f7_post_diagnosis_smoke/summary.json` |

---

## 4–11. Слои метрик, gold, данные, ID, F2–F6, F7

Содержательно без изменений к предыдущей версии: см. прежние разделы про substring vs chunking, `RerankerWrappingRetriever` + `fetch_k` в F7, несовпадение id fixture vs BM25, гипотезы по русскому домену и cross-encoder MS MARCO.

**Дополнение:** worst-queries таблица показывает для худших строк, попадает ли **verbatim** `references[0]` в текст **top-1** Chroma (колонка `ref in top1?`) — быстрый разделитель metric-limited vs retriever-limited.

---

## 12. Слой 8 — внешний mini Q/A

- Источник: **rajpurkar/squad** train split (80 строк), лицензия **CC BY-SA 4.0**.
- Скрипт материализации: [eval/diagnostics/materialize_sanity_squad_smoke.py](diagnostics/materialize_sanity_squad_smoke.py).
- Реестр коллекции (изолированное имя): [eval/config/embedding_bases_sanity_external.yaml](config/embedding_bases_sanity_external.yaml) → `sanity_external_squad_mpnet`.
- Индексация: `DATA/runs/phase_b_index_full/20260513_210612/manifest.json`.

---

## 13. Верификация (команды)

```text
python -m pytest tests/unit/test_retriever_evaluator.py tests/unit/test_reranker_wrapping_retriever.py -q
python -m pytest tests/integration/test_retrieval_recovery_report_smoke.py -q
python -m eval.reporting.build_retrieval_recovery_report
```

---

## 14. Next steps

1. Полная матрица B3 по **всем** `base_id` из YAML (не только `full_bge_m3`) — долго, планировать GPU.  
2. Расширить worst-queries скриптом: проверка «reference ∈ любой из top-20» через BM25 или экспорт JSONL.  
3. Полный F7 на 101 вопрос с trace, после выбора реранкера под русский текст.  
4. Answer-level eval при необходимости.

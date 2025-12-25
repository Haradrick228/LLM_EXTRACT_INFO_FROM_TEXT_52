# Eval артефакты

## Датасеты
- `eval_set.jsonl` — 101 вопрос без `answer_ref`.
- `eval_set_with_answer_ref.jsonl` — 101 вопрос с `answer_ref` (основной eval). Поле `references` — список релевантных чанков/файлов, на них считаются метрики поиска.
- `eval_set_with_answer_ref_sample20.jsonl` — 20 вопросов для быстрых прогонов/локальных моделей.
- `corpus_sample.jsonl` — 500 документов из Chroma (выборка для ручной проверки).

## Скрипты
- `run_retrieval_eval.py` — метрики поиска: recall@k, precision@k (k=5/10 по умолчанию), MRR, nDCG@3.
  - nDCG: бинарная релевантность по `references`, DCG с делителем `log2(rank+1)`, ранжирование — по скору эмбеддера/реранкера.
  - precision/recall считают совпадение строки чанка с любым `references` (регистр игнорируется, подстрока допустима).
- `run_answer_eval.py` — оценка ответов по `answer_ref` и `references`.
- `export_corpus_sample.py` — выгрузка `corpus_sample.jsonl`.
- `reindex_clean.py` — реиндексация `default_clean` с альтернативным эмбеддером.

## Результаты
- `rerank_results.md` — таблица сравнений реранкеров/конфигураций.
- `answer_eval_results.jsonl` и `answer_eval_predictions.jsonl` — прогон 101 вопрос (DeepSeek API).
- `answer_eval_results_ollama_sample20.jsonl` — прогон 20 вопросов на `qwen2.5:7b-instruct` через Ollama.
- `ANSWER_EVAL_SUMMARY.md` — краткая сводка ответных метрик и ссылки на исходные файлы.
- `notebooks/eval_report.ipynb` — обзорный ноутбук: загрузка результатов, визуализация метрик, примеры «вопрос → топ-чанки → ответ».

## Удалено/не используется
- `answer_eval_predictions_sample.jsonl` — битый дубликат, больше не используем.

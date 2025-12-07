# Eval артефакты (человечно)

## Датасеты
- `eval_set.jsonl` — 101 вопрос без `answer_ref`.
- `eval_set_with_answer_ref.jsonl` — 101 вопрос с заполненным `answer_ref` (основной eval).
- `eval_set_with_answer_ref_sample20.jsonl` — 20 вопросов для быстрых прогонов/локальных моделей.
- `corpus_sample.jsonl` — 500 документов из Chroma (выборка для ручной проверки).

## Скрипты
- `run_retrieval_eval.py` — метрики поиска (recall@k, MRR, nDCG).
- `run_answer_eval.py` — оценка ответов по `answer_ref` и `references`.
- `export_corpus_sample.py` — выгрузка `corpus_sample.jsonl`.
- `reindex_clean.py` — реиндексация `default_clean` с альтернативным эмбеддером.

## Результаты
- `rerank_results.md` — таблица сравнений реранкеров.
- `answer_eval_results.jsonl` и `answer_eval_predictions.jsonl` — прогон 101 вопрос (DeepSeek API).
- `answer_eval_results_ollama_sample20.jsonl` — прогон 20 вопросов на `qwen2.5:7b-instruct` через Ollama.
- `ANSWER_EVAL_SUMMARY.md` — краткая сводка ответных метрик.

## Удалено
- `answer_eval_predictions_sample.jsonl` — битый дубликат, больше не используем.

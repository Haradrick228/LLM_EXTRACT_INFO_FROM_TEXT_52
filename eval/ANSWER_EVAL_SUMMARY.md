# Сводка оценки ответов (Answer Eval)

Текущая конфигурация RAG: Chroma `default_clean` (paraphrase-multilingual-mpnet-base-v2) + rerank `all-MiniLM-L6-v2`, fetch_k=40, k=8.

## DeepSeek API — основной сет (101 вопрос)
- avg_refs_sim: 0.5998
- avg_ref_sim: 0.0886
- Файлы: `eval/answer_eval_results.jsonl`; снимок предсказаний — `archive/eval_research/snapshots/answer_eval_predictions.jsonl`.

## Ollama `qwen2.5:7b-instruct` (20 вопросов)
- avg_refs_sim: 0.5454
- avg_ref_sim: 0.0982
- Файл: `eval/answer_eval_results_ollama_sample20.jsonl`.

## DeepSeek API — ручной gold-набор (50 вопросов)
- avg_refs_sim: 0.5327
- avg_ref_sim: 0.5327
- Файл: `eval/answer_eval_results_gold.jsonl` (датасет `eval/eval_set_gold_manual.jsonl`).

## Ollama `qwen2.5:7b-instruct` — ручной gold-набор (50 вопросов)
- avg_refs_sim: 0.5327
- avg_ref_sim: 0.5327
- Файл: `eval/answer_eval_results_gold_ollama.jsonl` (тот же датасет `eval/eval_set_gold_manual.jsonl`).

## Примечания к методике
- Оценка ответов: косинусное сходство LLM-ответа с `answer_ref` и с лучшим из `references` (эмбеддер `all-MiniLM-L6-v2`).
- Метрики поиска (recall@k, precision@k, MRR, nDCG@3) считаются в `run_retrieval_eval.py`; релевантные фрагменты — из разметки `references`.
- nDCG: скидка 1/log2(rank+1), релевантность = 1 для размеченных фрагментов, иначе 0.

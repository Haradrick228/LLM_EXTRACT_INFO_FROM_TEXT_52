# Итоги оценки ответов

База: Chroma `default_clean` (paraphrase-multilingual-mpnet-base-v2) + би-реранкер `all-MiniLM-L6-v2`, fetch_k=40, k=8.

### DeepSeek API (101 вопросов)
- avg_refs_sim: 0.5998  
- avg_ref_sim: 0.0886  
- Файлы: `eval/answer_eval_results.jsonl`, `eval/answer_eval_predictions.jsonl`.

### Ollama `qwen2.5:7b-instruct` (20 вопросов)
- Контекст тот же, LLM через локальный OpenAI-совместимый endpoint.  
- avg_refs_sim: 0.5454  
- avg_ref_sim: 0.0982  
- Файл: `eval/answer_eval_results_ollama_sample20.jsonl`.

Что дальше
- Оставляем `default_clean` + `all-MiniLM-L6-v2` как базовую связку.
- Пробуем другие би-реранкеры/эмбеддеры на том же eval без переиндексации.

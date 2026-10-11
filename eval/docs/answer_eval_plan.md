# План экспериментов с ответами

Цель: измерить опору на контекст (similarity к `answer_ref`/`references`) для разных LLM и реранкеров.

Исходные данные
- `eval_set_with_answer_ref.jsonl` (101) и `eval_set_with_answer_ref_sample20.jsonl` (20).
- Контекст: Chroma `default_clean` + rerank `all-MiniLM-L6-v2` (fetch_k=40, k=8).
- Скрипт: `eval/run_answer_eval.py` → `answer_eval_results*.jsonl`.

Что уже сделано
- DeepSeek API на 101 вопросе (сводка в `ANSWER_EVAL_SUMMARY.md`).
- Ollama `qwen2.5:7b-instruct` на 20 вопросах (локальный прогон, метрики там же).

Дальше
1) Пробуем новые би-реранкеры/эмбеддеры на тех же датасетах без переиндексации.
2) При необходимости — другие LLM через OpenAI-совместимый API, фиксируем метрики в новых файлах `answer_eval_results_*.jsonl`.
3) В summary оставляем только краткие выводы и лучшую конфигурацию.

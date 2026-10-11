# Результаты поиска (Chroma `default_clean`)

Запуск: `python eval/run_retrieval_eval.py --eval-file eval/eval_set.jsonl`.

| Конфигурация                                   | fetch_k / k | recall@5 | recall@10 | MRR    | nDCG@3 | Комментарий                    |
|-----------------------------------------------|-------------|----------|-----------|--------|--------|--------------------------------|
| bi-rerank `all-MiniLM-L6-v2`                  | 40 / 8      | 0.0198   | 0.0198    | 0.0198 | 0.0280 | Лучшее среди протестированных  |
| базовый embed (paraphrase-multilingual)       | 50 / 5-10   | 0.0099   | 0.0198    | 0.0142 | 0.0091 | Без доп. реранка               |
| cross-encoder `ms-marco-MiniLM-L-6-v2`        | 40 / 8      | 0.0050   | 0.0050    | 0.0106 | 0.0099 | Сильно хуже                    |
| cross-encoder `ms-marco-electra-base`         | 20 / 8      | 0.0000   | 0.0050    | 0.0022 | 0.0000 | Провал                         |
| bi-rerank `all-MiniLM-L6-v2` на `default_allminilm` | 50 / 5-10   | 0.0050   | 0.0050    | 0.0099 | 0.0121 | Переиндексация ухудшила        |

Выводы
- Держим `default_clean` + `all-MiniLM-L6-v2` (bi-encoder), fetch_k=40, k=8 как базовую связку.
- Cross-encoder варианты хуже на всех метриках.
- Коллекция `default_allminilm` дала хуже recall — не используем.
- Дальше пробуем другие би-реранкеры/эмбеддеры на том же eval без переиндексации контента.

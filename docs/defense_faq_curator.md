# FAQ — типичные вопросы куратора (пром. защита / защита)

## Как размечали чанки для recall@k?

Gold `eval/datasets/eval_set_gold_v2.jsonl`: 101 вопрос, поля `references` (цитаты из документов) и `answer_ref`.  
Hit для **substring** = эталонная фраза входит в текст чанка (без учёта регистра).  
Скрипт: `eval/retrieval/run_retrieval_eval.py`.

## Почему recall@10 = 0.02 на старте?

Baseline **mpnet + MiniLM** на **full corpus** (61 543 чанка), не на fixture.  
2% = 0.02. После смены embedder/reranker на full corpus max **6.4%** (canonical 7×3 matrix).

## Почему в dashboard было 64%?

Ошибка eval setup: прогон фактически на fixture (~101 чанк), подписанный как full corpus.  
Исправлено; canonical report — единственный источник для full corpus. См. чат 31.05.2026.

## avg_ref_sim vs avg_refs_sim?

| Метрика | Сравниваем с |
|---------|----------------|
| **avg_ref_sim** | `answer_ref` — эталонный ответ |
| **avg_refs_sim** | лучший из `references` — фрагмент контекста |

Оба — cosine эмбеддингов (mpnet/MiniLM в ранних прогонах).  
Источник: `eval/docs/ANSWER_EVAL_SUMMARY.md`.

## Почему recall низкий, а avg_ref_sim ~0.66?

1. **~59%** вопросов — class A (эталон не в корпусе как substring) → потолок substring recall.  
2. **semantic** recall@10 до **71%** — «по смыслу» retrieval лучше.  
3. LLM (DeepSeek) генерирует связный ответ даже при неидеальном retrieval.

## Что такое матрица 7×3?

**7** embedder-индексов × **3** reranker-профиля = **21** конфигурация на 101 вопрос × 61 543 чанка.  
Не путать с fixture (101 чанк) и не путать с F2 hybrid (другой pipeline, без reranker из B3).

## Prod vs eval конфиг?

- **Eval canonical:** fetch_k=40, matrix profiles из `rag_profiles.yaml`.  
- **RagService defaults (без env):** fetch_k=**20**, top_k=**5** (`core/rag/service/rag_service.py`).  
- **Рекомендация prod:** `RAG_PROFILE` → USER-bge-m3 + BGE reranker (protocol `production_recommendation`).

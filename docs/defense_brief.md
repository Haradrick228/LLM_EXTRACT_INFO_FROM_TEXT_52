# Defense brief — RAG для БФ «Вклад в будущее» (14 слайдов)

Источник цифр: `eval/FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md` v2.  
Матрица: **7×3** = 7 embedders × 3 rerankers = **21** конфигурация.

---

## Слайд 1 — Титульный

**Тема:** Применение LLM для извлечения информации из текстов  
**Подзаголовок:** RAG-ассистент для БФ «Вклад в будущее» + исследование 7×3  
**Студент:** Серов Арсений · **Куратор:** Малюшитский Кирилл · **2025–2026**

---

## Слайд 2 — Проблема и цель

- БФ «Вклад в будущее», books.vbudushee.ru
- Проблема: сотни PDF, ручной поиск не масштабируется
- RAG: поиск фрагментов → ответ LLM по контексту
- Deliverable: Telegram-бот + API + исследование 7×3 на 61 543 фрагментах

---

## Слайд 3 — Пользователи

- Педагоги, родители, сотрудники фонда
- Инклюзивная среда, дети 5–18, ИИ/цифровые курсы
- Пример: «Школа возможностей»; abstain вне базы

---

## Слайд 4 — Данные

- Программы: «Развивающая среда», «Школа возможностей», Хрестоматия
- 973 док. → 61 543 чанка, 650/250
- EDA: ru 90.3%, dup 1.34%

---

## Слайд 5 — Архитектура

Telegram/REST :8001 → RagService :8000 → Chroma+BM25 → BGE reranker → LLM → PostgreSQL  
Eval: frozen protocol, fetch_k=40, 7×3

---

## Слайд 6 — Исследование 7×3

7 embedder (mpnet, BGE-M3, FRIDA, E5, Jina, default_clean, USER-bge-m3)  
3 reranker (MiniLM, BGE, Jina)  
+ BM25, hybrid, HyDE, multi-vector

---

## Слайды 7–8 — Чекпоинты

CP2 EDA · CP3 baseline ~2% · CP5 индексы · CP4 бот · CP6 canonical · CP7 traces

---

## Слайд 9 — Методология

Gold 101 · substring recall@10 · avg_ref_sim · ~59% class A · SQuAD 82.5% · shuffle 6.4%→3%

---

## Слайд 10 — Результаты

Max 6.4% · Prod USER-bge+BGE 5.4% · Best answer 0.66 · Не использовать 64% fixture

---

## Слайд 11 — Скорость

E2E 101 Q: avg 1.9s · median 1.7s · p95 3.0s · max 8.0s (USER-bge + BGE + qwen2.5)  
Embedder MPS medium: USER-bge 0.046s / 1370 MB · FRIDA 0.204s / 3141 MB  
HTML: rag_gold_101_full.html · embedding_benchmark_nagruzka_2026_05.html

---

## Слайд 12 — Top комбинаций

Таблица top-8 (читаемая); полная матрица в HTML

---

## Слайд 13 — Демо

Telegram + пример Q&A «Школа возможностей»

---

## Слайд 14 — Выводы

Продукт для фонда + исследование + честная методология + воспроизводимость

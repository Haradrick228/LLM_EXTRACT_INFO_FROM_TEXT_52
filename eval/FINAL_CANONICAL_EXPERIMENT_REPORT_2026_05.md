# Финальное испытание поиска по полному корпусу документов

*Отчёт v2, собран: 2026-06-02 11:21 UTC*

## Содержание

1. [Обзор](#1-обзор)
2. [Итоги](#2-итоги)
3. [Как устроен RAG](#3-как-устроен-rag)
4. [Эксперимент](#4-эксперимент)
5. [Словарь метрик](#5-словарь-метрик)
6. [Результаты](#6-результаты)
7. [Проверки доверия](#7-проверки-доверия)
8. [Ответы LLM](#8-ответы-llm)
9. [Приложение](#9-приложение)

---

## 1. Обзор

Мы проверили, **насколько хорошо система находит нужные фрагменты** в большой базе (**61,543** чанков), по **101** контрольным вопросам. Сравнили **7** моделей эмбеддинга и **3** reranker — **21** комбинаций.

Матрица **7×3**: в каждой ячейке измерен **embedder + reranker** (dense top-40 → rerank). Отдельного «только retrieval» в прогоне нет — см. **[раздел 2](#2-итоги)** (вердикт: embedder, reranker, комбинация для прода, answer-eval).

> У ~**59%** вопросов эталон **не встречается** в корпусе дословно (класс A) — потолок substring recall ограничен данными.

> Ответ LLM (`avg_ref_sim`) — отдельный прогон, раздел 8; не смешивать с Recall@10 по чанкам.

---

## 2. Итоги

101 вопросов · 61,543 чанков · матрица 21 ячеек (6 embedder × 3 reranker) · fetch_k=40.

Каждая ячейка = **dense retrieval (embedder)** → **rerank** → метрики по чанкам. Retrieval без rerank отдельно не снимали.

### Ключевая метрика

**`substring::recall@10`** — substring Recall@10 (дословный hit чанка в top-10). По ней сравниваем embedder, reranker и итоговую пару.

### Вердикт

| Что выбираем | ID | Показатель | Почему |
| --- | --- | --- | --- |
| Embedder | full_frida | max 6.4%, mean 5.6% | Лучший по max/mean `substring::recall@10` среди 6 индексов. |
| Reranker | rerank_bge_v2_m3 | max 6.4%, mean 4.7% | Лучший по max/mean среди 3 reranker; при max часто с `full_frida`. |
| Комбинация (retrieval) | full_user_bge_m3 + rerank_jina_v3 / full_frida + rerank_jina_v3 / full_frida + rerank_bge_v2_m3 | 6.4% | Максимум по ключевой метрике в матрице 6×3 (полный пайплайн). |
| **Брать в прод** | **full_user_bge_m3 + rerank_bge_v2_m3** | R@10 5.4% | USER-bge-m3 — быстрый индекс; BGE reranker в паре. Answer-eval по этой паре — см. таблицу ниже. |

> **Embedder:** Лучший max 6.4% по substring::recall@10 (mean 5.6% по 3 reranker). Отрыв max от 2-го embedder: —.

> **Reranker:** Лучший max 6.4% по substring::recall@10 (mean 4.7% по 6 embedder). Отрыв max от 2-го reranker: —.

### Точность ответа (BGE-M3)

**full_bge_m3 + rerank_bge_v2_m3** — R@10 5.4%, avg_ref_sim **0.660**. BAAI/bge-m3 (не USER-bge-m3): лучший avg_ref_sim на answer-eval — если важнее точность ответа LLM, а не скорость индекса.

**token_f1 Recall@10:** лидер `full_bge_m3` + `rerank_bge_v2_m3` — 6.9% (другая метрика, не ключевая).

**semantic Recall@10:** лидер `full_jina_embeddings_v3` + `rerank_bge_v2_m3` — 71.3% (другая метрика, не ключевая).

### Рейтинг embedder (ключевая метрика)

| # | embedder | mean | max | reranker @ max |
| --- | --- | --- | --- | --- |
| 1 | full_frida | 5.6% | 6.4% | rerank_bge_v2_m3 |
| 2 | full_user_bge_m3 | 5.1% | 6.4% | rerank_jina_v3 |
| 3 | full_e5_large_instruct | 4.1% | 5.9% | rerank_bge_v2_m3 |
| 4 | full_jina_embeddings_v3 | 3.5% | 5.9% | rerank_bge_v2_m3 |
| 5 | full_bge_m3 | 4.1% | 5.4% | rerank_bge_v2_m3 |
| 6 | full_mpnet | 1.5% | 2.0% | rerank_bge_v2_m3 |
| 7 | full_default_clean | 1.5% | 2.0% | rerank_bge_v2_m3 |

### Рейтинг reranker (ключевая метрика)

| # | reranker_id | название | mean | max | embedder @ max |
| --- | --- | --- | --- | --- | --- |
| 1 | rerank_bge_v2_m3 | BGE reranker | 4.7% | 6.4% | full_frida |
| 2 | rerank_jina_v3 | Jina reranker | 4.3% | 6.4% | full_frida |
| 3 | baseline_mpnet_rerank | MiniLM (базовый rerank) | 1.8% | 4.0% | full_frida |

### Топ комбинаций (ключевая метрика + answer если есть)

| # | embedder | reranker | R@10 | avg_ref_sim | N |
| --- | --- | --- | --- | --- | --- |
| 1 | full_user_bge_m3 | rerank_jina_v3 | 6.4% | — | — |
| 2 | full_frida | rerank_jina_v3 | 6.4% | 0.640 | 101 |
| 3 | full_frida | rerank_bge_v2_m3 | 6.4% | 0.650 | 101 |
| 4 | full_jina_embeddings_v3 | rerank_bge_v2_m3 | 5.9% | 0.632 | 101 |
| 5 | full_e5_large_instruct | rerank_jina_v3 | 5.9% | 0.644 | 101 |
| 6 | full_e5_large_instruct | rerank_bge_v2_m3 | 5.9% | 0.643 | 101 |
| 7 | full_user_bge_m3 | rerank_bge_v2_m3 | 5.4% | — | — |
| 8 | full_bge_m3 | rerank_bge_v2_m3 | 5.4% | 0.660 | 101 |

### Проверка ответа LLM (отдельный прогон)

На каждый из 101 вопросов: `RagService.answer` → **avg_ref_sim** (cosine эмбеддингов ответа и `answer_ref`). Это **не** Recall@10 по чанкам.

| # | Модель | Reranker | Retrieval R@10 | avg_ref_sim | N | Статус |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | BGE-M3 (bge-m3) | BGE reranker | 5.4% | 0.660 | 101 | OK |
| 2 | BGE-M3 (bge-m3) | Jina reranker | 4.5% | 0.655 | 101 | OK |
| 3 | FRIDA | BGE reranker | 6.4% | 0.650 | 101 | OK |
| 4 | E5-large-instruct (multilingual-e5-large-instruct) | Jina reranker | 5.9% | 0.644 | 101 | OK |
| 5 | E5-large-instruct (multilingual-e5-large-instruct) | BGE reranker | 5.9% | 0.643 | 101 | OK |
| 6 | FRIDA | Jina reranker | 6.4% | 0.640 | 101 | OK |
| 7 | Jina-embeddings-v3 (jina-embeddings-v3) | BGE reranker | 5.9% | 0.632 | 101 | OK |
| 8 | FRIDA | MiniLM (базовый rerank) | 4.0% | 0.615 | 101 | OK |

---

## 3. Как устроен RAG

```
Корпус (61k чанков) → 7 индексов Chroma (разные embedder)
       ↓
Вопрос → dense top-40 → rerank → топ-5/10/20
       ↓
Метрики hit: substring | token_f1 | semantic
       ↓
(отдельно) LLM answer → cosine(ответ, answer_ref)
```

| Этап | Что происходит | Что измеряли в этом отчёте |
|------|----------------|----------------------------|
| Индекс | Чанки + эмбеддинг в Chroma | 6+1 коллекций, одинаковый корпус |
| Retrieval | Векторный поиск top-40 | Время, кандидаты в trace |
| Rerank | Cross-encoder пересортировка | 3 профиля reranker |
| Оценка чанков | Hit в top-k | Recall, MRR, nDCG (3 режима) |
| Ответ LLM | RagService.answer | avg_ref_sim (FRIDA×3 + gap×5 + USER-bge-m3×3) |

---

## 4. Эксперимент

### Подготовка корпуса

~61,543 фрагментов текста экспортированы в JSONL и проиндексированы в **7 отдельных Chroma-коллекциях** (по одной модели эмбеддинга на коллекцию).

### Контрольные вопросы

**101** вопросов из `eval/datasets/eval_set_gold_v2.jsonl` — у каждого есть эталонная цитата (`references`) для проверки поиска.

### Сравнение всех комбинаций

Для каждой пары «модель поиска × пересортировка» (**7×3=21** вариантов) на каждый вопрос: векторный поиск **40** кандидатов → пересортировка → проверка, попал ли эталон в топ-5, 10 или 20.

### Объём прогонов

**2,121** отдельных оценок поиска (101 вопросов × 21 вариантов). Текст ответа нейросети на этом этапе не генерировали — только поиск.

### Проверки доверия

Отдельно: SQuAD sanity, neg-control с перетасованными эталонами, лестница A–D, зонд semantic-порога.

### Оценка ответов LLM

Для **8** конфигов (FRIDA×3 + строки 3–7 top-8) — отдельный прогон: `RagService.answer` + avg_ref_sim vs `answer_ref`.

| Параметр | Значение |
|----------|----------|
| fetch_k (до rerank) | **40** |
| k для метрик | 5, 10, 20 |
| Режимы hit | substring, token_f1, semantic (0.62) |

### Модели поиска

| Модель | Индекс |
|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | full_mpnet |
| BGE-M3 (bge-m3) | full_bge_m3 |
| FRIDA | full_frida |
| E5-large-instruct (multilingual-e5-large-instruct) | full_e5_large_instruct |
| Jina-embeddings-v3 (jina-embeddings-v3) | full_jina_embeddings_v3 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | full_default_clean |
| USER-bge-m3 | full_user_bge_m3 |

### Reranker

| Reranker | Роль |
|---|---|
| MiniLM (базовый rerank) | пересортировка 40 кандидатов |
| BGE reranker | пересортировка 40 кандидатов |
| Jina reranker | пересортировка 40 кандидатов |

---

## 5. Словарь метрик

| Метрика | Что это | Как считали | Когда смотреть | Не сравнивать с |
|---|---|---|---|---|
| Recall@k | Доля вопросов (из 101), где хотя бы один из k лучших найденных фрагментов считается «попавшим» по выбранному критерию. | Для каждого вопроса: 1, если hit в топ-k, иначе 0. Итог — среднее по 101 вопросу. | Главный показатель «нашли ли эталон в выдаче». Смотрим k=5, 10, 20. | Нельзя сравнивать recall между разными fetch_k или разными корпусами. |
| Precision@k | Средняя доля релевантных фрагментов среди k лучших (по тому же критерию hit). | Считается per-query, затем усредняется по 101 вопросу. | Показывает «засорённость» топа нерелевантными чанками. | — |
| MRR (Mean Reciprocal Rank) | Насколько высоко в списке стоит первый правильный фрагмент. | Для вопроса: 1/rank первого hit, или 0 если hit нет. MRR — среднее по 101 вопросу. | Важно, когда важен порядок: лучше hit на 1-й позиции, чем на 10-й при том же recall. | — |
| nDCG@3 | Качество ранжирования в топ-3 с учётом позиции релевантного чанка. | Normalized DCG@3 по списку кандидатов после пайплайна. | Дополнительный сигнал порядка в коротком топе. | — |
| Режим substring (дословно) | Чанк «попал», если эталонная фраза из gold — подстрока текста чанка (или наоборот), без учёта регистра. | hit = reference.lower() in chunk.lower() (или обратное включение). | Жёсткий критерий для отчёта: основная Recall@10 ~6.4% — именно substring. | Semantic recall и avg_ref_sim ответа LLM — другие шкалы. |
| Режим token_f1 | Мягче substring: пересечение токенов эталона и чанка (F1 по токенам). | token_overlap_f1_hit между reference и document_text. | Когда эталон разрезан чанкованием, но часть слов совпадает. | Не смешивать с substring % в одной «оценке успеха». |
| Режим semantic (поиск) | Чанк попал, если косинус эмбеддингов чанка и эталона ≥ порога 0.62 (модель paraphrase-multilingual-mpnet-base-v2). | cosine(embed(chunk), embed(reference)) ≥ 0.62 | Диагностика «по смыслу» на этапе retrieval. Цифры обычно выше substring. | 6.4% substring; avg_ref_sim ответа LLM (~64%). |
| SQuAD: Recall@10 (внешний контроль) | Отдельный «лёгкий» прогон: ~80 вопросов из SQuAD и маленький корпус, где правильный ответ дословно лежит в чанке. Проверяем, что поиск вообще работает. | Тот же substring::recall@10, но на external_sanity, не на 61k корпусе. | Sanity-check pipeline (порог PASS ≥ 50%; у нас ~82.5%). | Цифры полного корпуса 101×61k. |
| Негативный контроль (перетасованные эталоны) | Те же вопросы, но эталоны перепутаны с другими вопросами. Метрика должна упасть — иначе оценка «рисует» успех. | Честный gold vs shuffled_questions_gold_v2.jsonl, та же ячейка FRIDA+Jina. | Доверие к pipeline: ~6.4% → ~3.0% при shuffle. | — |
| Класс A (лестница метрик) | Эталонный текст не встречается ни в одном из ~61k экспортированных чанков как substring. Поиск по substring принципиально не может «попасть». | metrics_ladder_funnel по trace + corpus JSONL. | Объясняет потолок ~6% recall — не баг одной модели. | — |
| avg_ref_sim (ответ LLM) | Средняя косинусная близость эмбеддингов сгенерированного ответа и поля answer_ref. Это слой «качество ответа», не «нашли чанк». | mean(cosine(embed(answer), embed(answer_ref))) по 101 вопросу. | Отдельный прогон answer-eval для 8 конфигов (FRIDA×3 + top-8 gap×5; ~0.65 на лучшем). | 6.4% substring recall; semantic retrieval hit@k. |

---

## 6. Результаты

Лидер: **USER-bge-m3** + **Jina reranker** — substring Recall@10 = **6.4%**.

### Сводная таблица Recall@10 (%)

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.5% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 4.0% | **6.4%** | 6.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5% | 5.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 1.5% | 5.9% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.5% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |

### Топ-10 комбинаций

| # | Модель | Reranker | R@10 | R@20 |
|---|---|---|---|---|
| 1 | USER-bge-m3 | Jina reranker | 6.4% | 6.4% |
| 2 | FRIDA | Jina reranker | 6.4% | 7.4% |
| 3 | FRIDA | BGE reranker | 6.4% | 7.4% |
| 4 | Jina-embeddings-v3 (jina-embeddings-v3) | BGE reranker | 5.9% | 5.9% |
| 5 | E5-large-instruct (multilingual-e5-large-instruct) | Jina reranker | 5.9% | 5.9% |
| 6 | E5-large-instruct (multilingual-e5-large-instruct) | BGE reranker | 5.9% | 6.9% |
| 7 | USER-bge-m3 | BGE reranker | 5.4% | 5.4% |
| 8 | BGE-M3 (bge-m3) | BGE reranker | 5.4% | 5.4% |
| 9 | BGE-M3 (bge-m3) | Jina reranker | 4.5% | 4.5% |
| 10 | FRIDA | MiniLM (базовый rerank) | 4.0% | 5.4% |

### Recall@20 (substring, %)

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 1.0% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 5.4% | 7.4% | 7.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 3.0% | 6.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 2.0% | 5.9% | 4.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 1.0% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |


### MRR (substring)

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0025 | 0.0297 | 0.0142 |
| BGE-M3 (bge-m3) | 0.0086 | 0.0209 | 0.0442 |
| FRIDA | 0.0072 | 0.0236 | 0.0401 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0132 | 0.0224 | 0.0520 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0041 | 0.0208 | 0.0336 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0025 | 0.0297 | 0.0142 |
| USER-bge-m3 | 0.0103 | 0.0219 | 0.0663 |


---

## 7. Проверки доверия

| Проверка | Статус | Суть | Как запускали |
|---|---|---|---|
| SQuAD sanity | PASS | Recall@10 = 82.5% на ~80 лёгких вопросах | external_sanity, маленький корпус с дословным ответом |
| Neg-control | PASS | 6.4% → 3.0% | shuffled_questions_gold_v2.jsonl, та же FRIDA+Jina |
| Лестница A–D | PASS | ~59% класс A, ~6% класс D | metrics_ladder_funnel на BGE-M3 trace + corpus JSONL |
| Semantic probe | PASS | Калибровка порога 0.62 | semantic_threshold_probe на trace |

---

## 8. Ответы LLM

Отдельный прогон (не B3 retrieval matrix): на каждый из **101** вопросов — `RagService.answer` (Chroma + rerank + DeepSeek), затем **avg_ref_sim** — cosine(embed(ответ), embed(`answer_ref`)). Это **не** substring Recall@10 по чанкам.

#### Сводная (8 конфигов)

| # | Модель | Reranker | Retrieval R@10 | avg_ref_sim | N | Статус |
|---|---|---|---|---|---|---|
| 1 | BGE-M3 (bge-m3) | BGE reranker | 5.4% | 0.660 | 101 | OK |
| 2 | BGE-M3 (bge-m3) | Jina reranker | 4.5% | 0.655 | 101 | OK |
| 3 | FRIDA | BGE reranker | 6.4% | 0.650 | 101 | OK |
| 4 | E5-large-instruct (multilingual-e5-large-instruct) | Jina reranker | 5.9% | 0.644 | 101 | OK |
| 5 | E5-large-instruct (multilingual-e5-large-instruct) | BGE reranker | 5.9% | 0.643 | 101 | OK |
| 6 | USER-bge-m3 | BGE reranker | 5.4% | 0.642 | 101 | OK |
| 6 | FRIDA | Jina reranker | 6.4% | 0.640 | 101 | OK |
| 8 | Jina-embeddings-v3 (jina-embeddings-v3) | BGE reranker | 5.9% | 0.632 | 101 | OK |
| 9 | USER-bge-m3 | Jina reranker | 6.4% | 0.627 | 101 | OK |
| 10 | USER-bge-m3 | MiniLM (базовый rerank) | 3.5% | 0.625 | 101 | OK |
| 8 | FRIDA | MiniLM (базовый rerank) | 4.0% | 0.615 | 101 | OK |

#### FRIDA×3 (`_answer_top`)

| # | Модель | Reranker | Retrieval R@10 | avg_ref_sim | N | Статус |
|---|---|---|---|---|---|---|
| 6 | FRIDA | Jina reranker | 6.4% | 0.640 | 101 | OK |
| 3 | FRIDA | BGE reranker | 6.4% | 0.650 | 101 | OK |
| 8 | FRIDA | MiniLM (базовый rerank) | 4.0% | 0.615 | 101 | OK |

#### Top-8 gap, строки 3–7 (`_answer_top8_gap`)

| # | Модель | Reranker | Retrieval R@10 | avg_ref_sim | N | Статус |
|---|---|---|---|---|---|---|
| 8 | Jina-embeddings-v3 (jina-embeddings-v3) | BGE reranker | 5.9% | 0.632 | 101 | OK |
| 4 | E5-large-instruct (multilingual-e5-large-instruct) | Jina reranker | 5.9% | 0.644 | 101 | OK |
| 5 | E5-large-instruct (multilingual-e5-large-instruct) | BGE reranker | 5.9% | 0.643 | 101 | OK |
| 1 | BGE-M3 (bge-m3) | BGE reranker | 5.4% | 0.660 | 101 | OK |
| 2 | BGE-M3 (bge-m3) | Jina reranker | 4.5% | 0.655 | 101 | OK |

#### USER-bge-m3 supplement (`_answer_user_bge_m3`, 3×101)

| # | Модель | Reranker | Retrieval R@10 | avg_ref_sim | N | Статус |
|---|---|---|---|---|---|---|
| 10 | USER-bge-m3 | MiniLM (базовый rerank) | 3.5% | 0.625 | 101 | OK |
| 6 | USER-bge-m3 | BGE reranker | 5.4% | 0.642 | 101 | OK |
| 9 | USER-bge-m3 | Jina reranker | 6.4% | 0.627 | 101 | OK |

#### Расширенный набор (`_answer_diverse`, лидеры по метрикам)

*Не запускался.* Команда: `python -m eval.diagnostics.run_final_research_answer_diverse --b3-run-dir eval/runs/20260527_161527_b3_full_corpus_rerank`

---

## 9. Приложение

Полные матрицы 7×3 по **32** показателям, сгруппированы по режиму hit.

### Базовые (без префикса режима)

#### `recall@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.5% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 4.0% | 6.4% | 6.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5% | 5.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 1.5% | 5.9% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.5% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |

#### `recall@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 1.0% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 5.4% | 7.4% | 7.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 3.0% | 6.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 2.0% | 5.9% | 4.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 1.0% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |

#### `recall@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0% | 2.0% | 1.0% |
| BGE-M3 (bge-m3) | 1.0% | 2.5% | 3.5% |
| FRIDA | 0.0% | 3.0% | 5.9% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5% | 3.0% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0% | 3.0% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0% | 2.0% | 1.0% |
| USER-bge-m3 | 1.0% | 2.5% | 6.4% |

#### `mrr`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0025 | 0.0297 | 0.0142 |
| BGE-M3 (bge-m3) | 0.0086 | 0.0209 | 0.0442 |
| FRIDA | 0.0072 | 0.0236 | 0.0401 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0132 | 0.0224 | 0.0520 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0041 | 0.0208 | 0.0336 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0025 | 0.0297 | 0.0142 |
| USER-bge-m3 | 0.0103 | 0.0219 | 0.0663 |

#### `ndcg@3`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0297 | 0.0091 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0061 | 0.0319 |
| FRIDA | 0.0000 | 0.0061 | 0.0259 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0061 | 0.0061 | 0.0513 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0061 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0297 | 0.0091 |
| USER-bge-m3 | 0.0000 | 0.0061 | 0.0610 |

#### `precision@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0109 | 0.0040 |
| BGE-M3 (bge-m3) | 0.0109 | 0.0158 | 0.0228 |
| FRIDA | 0.0059 | 0.0218 | 0.0248 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0188 | 0.0109 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0030 | 0.0198 | 0.0089 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0109 | 0.0040 |
| USER-bge-m3 | 0.0149 | 0.0198 | 0.0257 |

#### `precision@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0099 | 0.0054 |
| BGE-M3 (bge-m3) | 0.0124 | 0.0208 | 0.0178 |
| FRIDA | 0.0109 | 0.0272 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0084 | 0.0203 | 0.0149 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0054 | 0.0168 | 0.0069 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0099 | 0.0054 |
| USER-bge-m3 | 0.0163 | 0.0213 | 0.0228 |

#### `precision@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0178 | 0.0059 |
| BGE-M3 (bge-m3) | 0.0059 | 0.0099 | 0.0198 |
| FRIDA | 0.0000 | 0.0099 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0040 | 0.0139 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0119 | 0.0158 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0178 | 0.0059 |
| USER-bge-m3 | 0.0059 | 0.0099 | 0.0356 |

### Дословное совпадение (substring)

#### `substring::mrr`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0025 | 0.0297 | 0.0142 |
| BGE-M3 (bge-m3) | 0.0086 | 0.0209 | 0.0442 |
| FRIDA | 0.0072 | 0.0236 | 0.0401 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0132 | 0.0224 | 0.0520 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0041 | 0.0208 | 0.0336 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0025 | 0.0297 | 0.0142 |
| USER-bge-m3 | 0.0103 | 0.0219 | 0.0663 |

#### `substring::ndcg@3`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0297 | 0.0091 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0061 | 0.0319 |
| FRIDA | 0.0000 | 0.0061 | 0.0259 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0061 | 0.0061 | 0.0513 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0061 | 0.0297 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0297 | 0.0091 |
| USER-bge-m3 | 0.0000 | 0.0061 | 0.0610 |

#### `substring::precision@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0109 | 0.0040 |
| BGE-M3 (bge-m3) | 0.0109 | 0.0158 | 0.0228 |
| FRIDA | 0.0059 | 0.0218 | 0.0248 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0050 | 0.0188 | 0.0109 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0030 | 0.0198 | 0.0089 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0109 | 0.0040 |
| USER-bge-m3 | 0.0149 | 0.0198 | 0.0257 |

#### `substring::precision@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0030 | 0.0099 | 0.0054 |
| BGE-M3 (bge-m3) | 0.0124 | 0.0208 | 0.0178 |
| FRIDA | 0.0109 | 0.0272 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0084 | 0.0203 | 0.0149 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0054 | 0.0168 | 0.0069 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0030 | 0.0099 | 0.0054 |
| USER-bge-m3 | 0.0163 | 0.0213 | 0.0228 |

#### `substring::precision@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0178 | 0.0059 |
| BGE-M3 (bge-m3) | 0.0059 | 0.0099 | 0.0198 |
| FRIDA | 0.0000 | 0.0099 | 0.0238 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0040 | 0.0139 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0119 | 0.0158 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0178 | 0.0059 |
| USER-bge-m3 | 0.0059 | 0.0099 | 0.0356 |

#### `substring::recall@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.5% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 4.0% | 6.4% | 6.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5% | 5.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 1.5% | 5.9% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.5% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |

#### `substring::recall@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 1.0% | 2.0% | 2.0% |
| BGE-M3 (bge-m3) | 2.5% | 5.4% | 4.5% |
| FRIDA | 5.4% | 7.4% | 7.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 3.0% | 6.9% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 2.0% | 5.9% | 4.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 1.0% | 2.0% | 2.0% |
| USER-bge-m3 | 3.5% | 5.4% | 6.4% |

#### `substring::recall@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0% | 2.0% | 1.0% |
| BGE-M3 (bge-m3) | 1.0% | 2.5% | 3.5% |
| FRIDA | 0.0% | 3.0% | 5.9% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.5% | 3.0% | 5.9% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0% | 3.0% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0% | 2.0% | 1.0% |
| USER-bge-m3 | 1.0% | 2.5% | 6.4% |

### Пересечение токенов (token_f1)

#### `token_f1::mrr`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0036 | 0.0484 | 0.0185 |
| FRIDA | 0.0026 | 0.0113 | 0.0032 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0021 | 0.0124 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0022 | 0.0149 | 0.0114 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |
| USER-bge-m3 | 0.0033 | 0.0435 | 0.0311 |

#### `token_f1::ndcg@3`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0495 | 0.0099 |
| FRIDA | 0.0000 | 0.0099 | 0.0000 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0000 | 0.0099 | 0.0198 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0099 | 0.0099 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |
| USER-bge-m3 | 0.0000 | 0.0396 | 0.0297 |

#### `token_f1::precision@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0010 | 0.0158 | 0.0069 |
| FRIDA | 0.0020 | 0.0030 | 0.0010 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0010 | 0.0030 | 0.0030 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0040 | 0.0020 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |
| USER-bge-m3 | 0.0010 | 0.0109 | 0.0040 |

#### `token_f1::precision@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0020 | 0.0079 | 0.0079 |
| FRIDA | 0.0015 | 0.0015 | 0.0015 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0015 | 0.0015 | 0.0015 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0015 | 0.0020 | 0.0020 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |
| USER-bge-m3 | 0.0020 | 0.0054 | 0.0045 |

#### `token_f1::precision@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0000 | 0.0000 | 0.0000 |
| BGE-M3 (bge-m3) | 0.0000 | 0.0297 | 0.0079 |
| FRIDA | 0.0020 | 0.0040 | 0.0020 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0000 | 0.0059 | 0.0059 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0000 | 0.0079 | 0.0040 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0000 | 0.0000 | 0.0000 |
| USER-bge-m3 | 0.0000 | 0.0198 | 0.0079 |

#### `token_f1::recall@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0% | 0.0% | 0.0% |
| BGE-M3 (bge-m3) | 1.0% | 6.9% | 4.0% |
| FRIDA | 1.0% | 2.0% | 1.0% |
| E5-large-instruct (multilingual-e5-large-instruct) | 1.0% | 2.0% | 2.0% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0% | 3.0% | 1.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0% | 0.0% | 0.0% |
| USER-bge-m3 | 1.0% | 5.9% | 3.0% |

#### `token_f1::recall@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0% | 0.0% | 0.0% |
| BGE-M3 (bge-m3) | 3.0% | 6.9% | 6.9% |
| FRIDA | 2.0% | 2.0% | 2.0% |
| E5-large-instruct (multilingual-e5-large-instruct) | 2.0% | 2.0% | 2.0% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 3.0% | 3.0% | 3.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0% | 0.0% | 0.0% |
| USER-bge-m3 | 3.0% | 5.9% | 5.0% |

#### `token_f1::recall@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.0% | 0.0% | 0.0% |
| BGE-M3 (bge-m3) | 0.0% | 5.9% | 3.0% |
| FRIDA | 1.0% | 1.0% | 1.0% |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.0% | 2.0% | 2.0% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.0% | 3.0% | 1.0% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.0% | 0.0% | 0.0% |
| USER-bge-m3 | 0.0% | 5.0% | 3.0% |

### Смысловая близость чанка (semantic)

#### `semantic::mrr`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3830 | 0.5692 | 0.4920 |
| BGE-M3 (bge-m3) | 0.3725 | 0.6079 | 0.5281 |
| FRIDA | 0.3402 | 0.5278 | 0.4814 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.3428 | 0.5831 | 0.4587 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.4289 | 0.6157 | 0.5285 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3784 | 0.5668 | 0.5029 |
| USER-bge-m3 | 0.4038 | 0.6216 | 0.5192 |

#### `semantic::ndcg@3`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3763 | 0.5639 | 0.4886 |
| BGE-M3 (bge-m3) | 0.3795 | 0.6037 | 0.5273 |
| FRIDA | 0.3270 | 0.5392 | 0.4966 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.3395 | 0.6035 | 0.4803 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.4389 | 0.6012 | 0.5491 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3677 | 0.5540 | 0.5034 |
| USER-bge-m3 | 0.3986 | 0.6209 | 0.5411 |

#### `semantic::precision@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3376 | 0.4178 | 0.3752 |
| BGE-M3 (bge-m3) | 0.3089 | 0.4020 | 0.3604 |
| FRIDA | 0.2604 | 0.3446 | 0.3079 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2634 | 0.3693 | 0.3020 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3267 | 0.4089 | 0.3762 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3376 | 0.4149 | 0.3772 |
| USER-bge-m3 | 0.3149 | 0.3960 | 0.3713 |

#### `semantic::precision@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3391 | 0.3748 | 0.3619 |
| BGE-M3 (bge-m3) | 0.3089 | 0.3441 | 0.3322 |
| FRIDA | 0.2535 | 0.2965 | 0.2748 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2688 | 0.3203 | 0.2777 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3203 | 0.3609 | 0.3485 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3396 | 0.3762 | 0.3609 |
| USER-bge-m3 | 0.3178 | 0.3663 | 0.3475 |

#### `semantic::precision@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 0.3129 | 0.4475 | 0.3980 |
| BGE-M3 (bge-m3) | 0.2990 | 0.4733 | 0.3921 |
| FRIDA | 0.2475 | 0.4040 | 0.3525 |
| E5-large-instruct (multilingual-e5-large-instruct) | 0.2495 | 0.4455 | 0.3307 |
| Jina-embeddings-v3 (jina-embeddings-v3) | 0.3327 | 0.4752 | 0.4059 |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 0.3129 | 0.4436 | 0.4040 |
| USER-bge-m3 | 0.3129 | 0.4733 | 0.4059 |

#### `semantic::recall@10`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 57.9% | 66.3% | 60.9% |
| BGE-M3 (bge-m3) | 62.9% | 67.3% | 61.9% |
| FRIDA | 55.9% | 60.4% | 62.4% |
| E5-large-instruct (multilingual-e5-large-instruct) | 57.4% | 66.3% | 60.4% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 61.4% | 71.3% | 68.8% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 58.9% | 65.3% | 61.9% |
| USER-bge-m3 | 62.9% | 67.3% | 64.4% |

#### `semantic::recall@20`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 64.4% | 71.8% | 67.8% |
| BGE-M3 (bge-m3) | 68.3% | 71.3% | 67.8% |
| FRIDA | 61.9% | 65.3% | 66.3% |
| E5-large-instruct (multilingual-e5-large-instruct) | 67.8% | 70.8% | 68.3% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 69.3% | 76.7% | 72.8% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 66.3% | 71.8% | 68.8% |
| USER-bge-m3 | 70.3% | 71.8% | 71.3% |

#### `semantic::recall@5`

| Модель поиска | MiniLM (базовый rerank) | BGE reranker | Jina reranker |
|---|---|---|---|
| mpnet (paraphrase-multilingual-mpnet-base-v2) | 49.0% | 60.4% | 56.4% |
| BGE-M3 (bge-m3) | 50.0% | 63.4% | 56.9% |
| FRIDA | 46.0% | 55.9% | 56.9% |
| E5-large-instruct (multilingual-e5-large-instruct) | 45.0% | 62.9% | 54.0% |
| Jina-embeddings-v3 (jina-embeddings-v3) | 54.5% | 66.3% | 63.9% |
| mpnet (legacy default index): paraphrase-multilingual-mpnet-base-v2 · Chroma «default» | 49.0% | 60.4% | 57.4% |
| USER-bge-m3 | 54.0% | 64.4% | 57.9% |

---

## Для разработчиков

- run_dir: `eval/runs/20260527_161527_b3_full_corpus_rerank`
- Протокол: `eval/config/final_research_protocol.yaml`
- HTML: `analysis/reports/final_canonical_experiment_2026_05.html`

```bash
python -m eval.reporting.emit_final_canonical_experiment_report --write-html
```

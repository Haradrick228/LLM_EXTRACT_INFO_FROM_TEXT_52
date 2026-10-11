# Full corpus retrieval — сводные таблицы (2026-05)

Источник цифр: автоматический отчёт [analysis/reports/retrieval_recovery_2026_05.html](../analysis/reports/retrieval_recovery_2026_05.html), собранный из `eval/runs/*` на момент генерации HTML. Датасет: **`eval/datasets/eval_set_gold_v2.jsonl`** (101 вопрос). Корпус: **61 543** чанка (`DATA/indexes/bm25/corpus_full_20260512/meta.json`).

| Поле | Значение |
|------|-----------|
| `gold_sha256` | `d2d35200b634b92933989c8457f8bc6fe670b9edf7a4ab688ab8e3ae7ae50c96` |
| `BM25_DIR` | `DATA/indexes/bm25/corpus_full_20260512` |
| `chunking_label` (реестр) | `650_250` ([eval/config/embedding_bases_full_corpus.yaml](config/embedding_bases_full_corpus.yaml)) |

Run directories (по плану Phase B recovery):

| Фаза | Каталог `eval/runs/` |
|------|----------------------|
| F2 hybrid RRF sweep | `20260513_015456_f2_hybrid_rrf_sweep` |
| F3 query transform | `20260513_020722_f3_query_transform_sweep` |
| F4 BGE-M3 multi-vector | `20260513_020808_f4_bge_m3_multi_vector_sweep` |
| F7 final reranker matrix | `20260513_122704_f7_final_reranker_matrix` |
| B3 full corpus rerank (subset) | `20260513_210857_b3_full_corpus_rerank` (`full_bge_m3` × 3 matrix-профиля, `fetch_k=40`) |
| Gold dense + trace (full corpus) | `20260513_205851_gold_on_full_corpus_bge_m3` |
| Внешний SQuAD sanity | `20260513_sanity_external_squad_mpnet` |
| F7 smoke (12 вопросов) | `20260513_210704_f7_post_diagnosis_smoke` |

---

## F2 — Hybrid RRF (`rrf_k=60`), `fetch_k ∈ {40,100,200,400}`

Срез **fetch_k = 200**, режим **`hybrid_rrf`** (остальные комбинации — в HTML по `<details>`).

| `base_id` | recall@5 | recall@10 | recall@20 | precision@5 | precision@10 | mrr | ndcg@3 |
|-----------|----------|-----------|-----------|---------------|----------------|-----|--------|
| full_mpnet | 0.0347 | 0.0347 | 0.0545 | 0.0158 | 0.0139 | 0.0267 | 0.0161 |
| full_bge_m3 | 0.0248 | 0.0347 | 0.0545 | 0.0218 | 0.0188 | 0.0374 | 0.0358 |
| full_frida | 0.0198 | 0.0594 | 0.0644 | 0.0198 | 0.0218 | 0.0354 | 0.0297 |
| full_e5_large_instruct | 0.0644 | 0.0743 | 0.0743 | 0.0356 | 0.0267 | 0.0689 | 0.0704 |
| full_jina_embeddings_v3 | 0.0248 | 0.0347 | 0.0644 | 0.0198 | 0.0178 | 0.0385 | 0.0319 |
| full_default_clean | 0.0248 | 0.0248 | 0.0446 | 0.0139 | 0.0129 | 0.0242 | 0.0161 |

`dense_only` / `bm25_only` / другие `fetch_k` — в HTML-отчёте.

---

## F3 — Query transforms (Ollama `qwen2.5:7b-instruct`, `fetch_k=100`, `rrf_k=60`)

| `base_id` | dense_only R@10 | hyde R@10 | mqr R@10 | hyde_mqr_dense_bm25_rrf R@10 |
|-----------|-----------------|-----------|----------|-------------------------------|
| full_mpnet | 0.0198 | 0.0000 | 0.0050 | 0.0050 |
| full_bge_m3 | 0.0446 | 0.0149 | 0.0644 | 0.0347 |
| full_frida | 0.0644 | 0.0149 | 0.0396 | 0.0495 |
| full_e5_large_instruct | 0.0594 | 0.0248 | 0.0545 | 0.0495 |
| full_jina_embeddings_v3 | 0.0297 | 0.0050 | 0.0495 | 0.0198 |
| full_default_clean | 0.0198 | 0.0000 | 0.0099 | 0.0149 |

---

## F4 — BGE-M3 multi-vector (`candidate_top_k=100`, `top_k=20`)

| combo | recall@5 | recall@10 | recall@20 | mrr | ndcg@3 |
|-------|----------|-----------|-----------|-----|--------|
| dense_only | 0.0347 | 0.0446 | 0.0446 | 0.0432 | 0.0319 |
| dense_sparse | 0.0446 | 0.0545 | 0.0545 | 0.0452 | 0.0350 |
| dense_colbert | 0.0545 | 0.0644 | 0.0644 | 0.0470 | 0.0319 |
| balanced | 0.0545 | 0.0545 | 0.0644 | 0.0473 | 0.0350 |
| sparse_heavy | 0.0446 | 0.0545 | 0.0644 | 0.0462 | 0.0350 |

---

## F7 — Final reranker matrix

База: **hybrid_rrf** на `full_mpnet` + BM25 sidecar (`rrf_k=60`), как в `summary` HTML.

| reranker | recall@5 | recall@10 | recall@20 | mrr | ndcg@3 |
|----------|----------|-----------|-----------|-----|--------|
| `__no_rerank__` | 0.0347 | 0.0347 | 0.0545 | 0.0256 | 0.0161 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0050 | 0.0050 | 0.0050 | 0.0099 | 0.0099 |
| BAAI/bge-reranker-v2-m3 | 0.0000 | 0.0050 | 0.0149 | 0.0021 | 0.0000 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0347 | 0.0347 | 0.0545 | 0.0256 | 0.0161 |

---

## B3 — Full corpus rerank matrix (subset: `full_bge_m3` only)

Полный прогон по **всем** dense-слотам YAML: `python -m eval.experiments.replay_experiment run phase_b3_full_corpus_rerank_matrix`. Здесь — **одна** база `full_bge_m3`, `fetch_k=40`, все профили с `matrix: true` в [eval/config/rag_profiles.yaml](config/rag_profiles.yaml), артефакты: `eval/runs/20260513_210857_b3_full_corpus_rerank/`.

| profile | recall@5 | recall@10 | recall@20 | token_f1 R@10 | mrr | ndcg@3 |
|---------|----------|-----------|-----------|---------------|-----|--------|
| baseline_mpnet_rerank (`all-MiniLM-L6-v2`) | 0.0099 | 0.0248 | 0.0248 | 0.0099 | 0.0086 | 0.0000 |
| rerank_jina_v3 | 0.0347 | 0.0446 | 0.0446 | 0.0396 | 0.0442 | 0.0319 |
| rerank_bge_v2_m3 | 0.0248 | 0.0545 | 0.0545 | 0.0693 | 0.0209 | 0.0061 |

---

## Gold на full corpus (dense, без реранка, с trace)

Те же **101** вопроса `eval_set_gold_v2.jsonl`, коллекция **`corpus_phase_b_bge_m3`** (61 543 чанка), `fetch_k=100`, `k=5,10,20`, `--retrieval-trace-jsonl` → `eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/`.

| Метрика | substring | token_f1 |
|---------|-----------|----------|
| recall@5 | 0.0347 | 0.0297 |
| recall@10 | 0.0446 | 0.0396 |
| recall@20 | 0.0446 | 0.0693 |
| mrr | 0.0442 | 0.0187 |
| ndcg@3 | 0.0319 | 0.0099 |

Разбор худших строк: [RETRIEVAL_WORST_QUERIES_2026_05.md](RETRIEVAL_WORST_QUERIES_2026_05.md).

---

## Внешний sanity (SQuAD, изолированная коллекция)

- 80 параграфов из `rajpurkar/squad` → `DATA/runs/sanity_external_squad_smoke/corpus.jsonl` + `gold.jsonl` (CC BY-SA 4.0).
- Chroma: **`sanity_external_squad_mpnet`** (реестр [embedding_bases_sanity_external.yaml](config/embedding_bases_sanity_external.yaml)).
- Eval: `eval/runs/20260513_sanity_external_squad_mpnet/retrieval.json` — **substring recall@10 = 0.825** (token_f1 на коротких ответах ≈ 0 — ожидаемо).

---

## Контрольный smoke (fixture — не путать с full corpus)

Команда (коллекция **~101 чанк**, не 61k; `--max-rows 25` = только **первые 25** вопросов из gold):

`python -m eval.retrieval.run_retrieval_eval --eval-file eval/datasets/eval_set_gold_v2.jsonl --chroma-collection phase_b_gold_v2_bge_m3 --embed-model BAAI/bge-m3 --fetch-k 40 --k 5,10 --max-rows 25 --out-json eval/runs/20260513_sanity_fixture_bge_m3_smoke/retrieval.json`

Результат: высокий recall на фикстуре проверяет код и Chroma; **не** заменяет строку «Gold на full corpus» выше.

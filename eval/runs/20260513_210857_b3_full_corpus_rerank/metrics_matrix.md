# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_210857_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт).

## recall@5

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0099 | 0.0347 | 0.0248 |

## recall@10

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0248 | 0.0446 | 0.0545 |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0059 | 0.0198 | 0.0099 |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0109 | 0.0228 | 0.0158 |

## mrr

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0086 | 0.0442 | 0.0209 |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_jina_v3 | rerank_bge_v2_m3 |
|---|---|---|---|
| full_bge_m3 | 0.0000 | 0.0319 | 0.0061 |

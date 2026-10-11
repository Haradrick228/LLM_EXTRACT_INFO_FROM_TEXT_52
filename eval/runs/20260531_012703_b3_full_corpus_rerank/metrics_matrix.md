# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260531_012703_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт). При `--metric-modes substring,token_f1,semantic` появляются ключи вида `token_f1::recall@10`.

## recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0347 | 0.0545 | 0.0644 |

## recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0347 | 0.0545 | 0.0644 |

## recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0099 | 0.0248 | 0.0644 |

## mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0103 | 0.0219 | 0.0663 |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0000 | 0.0061 | 0.0610 |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0149 | 0.0198 | 0.0257 |

## precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0163 | 0.0213 | 0.0228 |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0059 | 0.0099 | 0.0356 |

## semantic::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.4038 | 0.6216 | 0.5192 |

## semantic::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.3986 | 0.6209 | 0.5411 |

## semantic::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.3149 | 0.3960 | 0.3713 |

## semantic::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.3178 | 0.3663 | 0.3475 |

## semantic::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.3129 | 0.4733 | 0.4059 |

## semantic::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.6287 | 0.6733 | 0.6436 |

## semantic::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.7030 | 0.7178 | 0.7129 |

## semantic::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.5396 | 0.6436 | 0.5792 |

## substring::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0103 | 0.0219 | 0.0663 |

## substring::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0000 | 0.0061 | 0.0610 |

## substring::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0149 | 0.0198 | 0.0257 |

## substring::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0163 | 0.0213 | 0.0228 |

## substring::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0059 | 0.0099 | 0.0356 |

## substring::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0347 | 0.0545 | 0.0644 |

## substring::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0347 | 0.0545 | 0.0644 |

## substring::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0099 | 0.0248 | 0.0644 |

## token_f1::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0033 | 0.0435 | 0.0311 |

## token_f1::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0000 | 0.0396 | 0.0297 |

## token_f1::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0010 | 0.0109 | 0.0040 |

## token_f1::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0020 | 0.0054 | 0.0045 |

## token_f1::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0000 | 0.0198 | 0.0079 |

## token_f1::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0099 | 0.0594 | 0.0297 |

## token_f1::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0297 | 0.0594 | 0.0495 |

## token_f1::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_user_bge_m3 | 0.0000 | 0.0495 | 0.0297 |

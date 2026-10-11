# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_114047_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт).

## recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0248 | 0.0099 |
| full_bge_m3 | 0.0099 | 0.0248 | 0.0347 |
| full_frida | 0.0099 | 0.0248 | 0.0594 |
| full_e5_large_instruct | 0.0099 | 0.0198 | 0.0594 |
| full_jina_embeddings_v3 | 0.0099 | 0.0198 | 0.0297 |
| full_default_clean | 0.0000 | 0.0248 | 0.0099 |

## recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0248 | 0.0198 |
| full_bge_m3 | 0.0198 | 0.0545 | 0.0446 |
| full_frida | 0.0198 | 0.0545 | 0.0644 |
| full_e5_large_instruct | 0.0198 | 0.0495 | 0.0594 |
| full_jina_embeddings_v3 | 0.0198 | 0.0594 | 0.0297 |
| full_default_clean | 0.0000 | 0.0248 | 0.0198 |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0238 | 0.0059 |
| full_bge_m3 | 0.0119 | 0.0099 | 0.0198 |
| full_frida | 0.0040 | 0.0099 | 0.0238 |
| full_e5_large_instruct | 0.0040 | 0.0079 | 0.0198 |
| full_jina_embeddings_v3 | 0.0040 | 0.0079 | 0.0158 |
| full_default_clean | 0.0000 | 0.0238 | 0.0059 |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0158 | 0.0040 |
| full_bge_m3 | 0.0129 | 0.0149 | 0.0228 |
| full_frida | 0.0030 | 0.0178 | 0.0248 |
| full_e5_large_instruct | 0.0069 | 0.0139 | 0.0109 |
| full_jina_embeddings_v3 | 0.0040 | 0.0158 | 0.0089 |
| full_default_clean | 0.0000 | 0.0158 | 0.0040 |

## mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0012 | 0.0396 | 0.0144 |
| full_bge_m3 | 0.0227 | 0.0206 | 0.0442 |
| full_frida | 0.0162 | 0.0220 | 0.0402 |
| full_e5_large_instruct | 0.0219 | 0.0196 | 0.0520 |
| full_jina_embeddings_v3 | 0.0147 | 0.0191 | 0.0336 |
| full_default_clean | 0.0011 | 0.0396 | 0.0143 |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0358 | 0.0091 |
| full_bge_m3 | 0.0160 | 0.0061 | 0.0319 |
| full_frida | 0.0061 | 0.0061 | 0.0259 |
| full_e5_large_instruct | 0.0121 | 0.0061 | 0.0513 |
| full_jina_embeddings_v3 | 0.0091 | 0.0061 | 0.0297 |
| full_default_clean | 0.0000 | 0.0396 | 0.0091 |

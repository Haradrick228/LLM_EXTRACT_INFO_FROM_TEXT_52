# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260512_161557_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт).

## recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.1250 | 0.1250 | 0.0000 |
| full_bge_m3 | 0.1250 | 0.0000 | 0.1250 |
| full_frida | 0.1250 | 0.0000 | 0.1250 |
| full_e5_large_instruct | 0.1250 | 0.1250 | 0.1250 |
| full_jina_embeddings_v3 | 0.0000 | 0.1250 | 0.1250 |
| full_default_clean | 0.1250 | 0.1250 | 0.0000 |

## recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.1250 | 0.1250 | 0.1250 |
| full_bge_m3 | 0.1250 | 0.1250 | 0.1250 |
| full_frida | 0.1250 | 0.1250 | 0.1250 |
| full_e5_large_instruct | 0.1250 | 0.1250 | 0.1250 |
| full_jina_embeddings_v3 | 0.0000 | 0.1250 | 0.1250 |
| full_default_clean | 0.1250 | 0.1250 | 0.1250 |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0250 | 0.0500 | 0.0000 |
| full_bge_m3 | 0.0750 | 0.0000 | 0.0500 |
| full_frida | 0.0750 | 0.0000 | 0.0500 |
| full_e5_large_instruct | 0.0250 | 0.0500 | 0.0500 |
| full_jina_embeddings_v3 | 0.0000 | 0.0500 | 0.0500 |
| full_default_clean | 0.0250 | 0.0500 | 0.0000 |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0250 | 0.0250 | 0.0125 |
| full_bge_m3 | 0.0500 | 0.0500 | 0.0875 |
| full_frida | 0.0500 | 0.0500 | 0.0500 |
| full_e5_large_instruct | 0.0250 | 0.0750 | 0.0250 |
| full_jina_embeddings_v3 | 0.0000 | 0.0750 | 0.0375 |
| full_default_clean | 0.0125 | 0.0250 | 0.0125 |

## mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0417 | 0.1250 | 0.0125 |
| full_bge_m3 | 0.0625 | 0.0179 | 0.1250 |
| full_frida | 0.1250 | 0.0179 | 0.0312 |
| full_e5_large_instruct | 0.1250 | 0.0312 | 0.1250 |
| full_jina_embeddings_v3 | 0.0060 | 0.0312 | 0.1250 |
| full_default_clean | 0.0625 | 0.1250 | 0.0125 |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0625 | 0.1250 | 0.0000 |
| full_bge_m3 | 0.1250 | 0.0000 | 0.1250 |
| full_frida | 0.1250 | 0.0000 | 0.0000 |
| full_e5_large_instruct | 0.1250 | 0.0000 | 0.1250 |
| full_jina_embeddings_v3 | 0.0000 | 0.0000 | 0.1250 |
| full_default_clean | 0.0789 | 0.1250 | 0.0000 |

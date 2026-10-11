# F3 query-transform sweep

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_020722_f3_query_transform_sweep`

fetch_k = `100`; RRF k (фьюжн) = `60`
LLM provider: `ollama` model `qwen2.5:7b-instruct`

## recall@5

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0099 | 0.0000 | 0.0050 | 0.0050 |
| full_bge_m3 | 0.0347 | 0.0149 | 0.0446 | 0.0297 |
| full_frida | 0.0594 | 0.0149 | 0.0297 | 0.0198 |
| full_e5_large_instruct | 0.0594 | 0.0149 | 0.0545 | 0.0495 |
| full_jina_embeddings_v3 | 0.0297 | 0.0050 | 0.0495 | 0.0198 |
| full_default_clean | 0.0099 | 0.0000 | 0.0099 | 0.0050 |

## recall@10

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0198 | 0.0000 | 0.0050 | 0.0050 |
| full_bge_m3 | 0.0446 | 0.0149 | 0.0644 | 0.0347 |
| full_frida | 0.0644 | 0.0149 | 0.0396 | 0.0495 |
| full_e5_large_instruct | 0.0594 | 0.0248 | 0.0545 | 0.0495 |
| full_jina_embeddings_v3 | 0.0297 | 0.0050 | 0.0495 | 0.0198 |
| full_default_clean | 0.0198 | 0.0000 | 0.0099 | 0.0149 |

## recall@20

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0198 | 0.0050 | 0.0198 | 0.0198 |
| full_bge_m3 | 0.0446 | 0.0297 | 0.0644 | 0.0446 |
| full_frida | 0.0743 | 0.0149 | 0.0545 | 0.0545 |
| full_e5_large_instruct | 0.0594 | 0.0248 | 0.0743 | 0.0644 |
| full_jina_embeddings_v3 | 0.0396 | 0.0198 | 0.0545 | 0.0198 |
| full_default_clean | 0.0198 | 0.0000 | 0.0198 | 0.0198 |

## precision@5

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0059 | 0.0000 | 0.0059 | 0.0020 |
| full_bge_m3 | 0.0198 | 0.0059 | 0.0218 | 0.0158 |
| full_frida | 0.0238 | 0.0040 | 0.0158 | 0.0099 |
| full_e5_large_instruct | 0.0198 | 0.0040 | 0.0317 | 0.0178 |
| full_jina_embeddings_v3 | 0.0158 | 0.0020 | 0.0218 | 0.0099 |
| full_default_clean | 0.0059 | 0.0000 | 0.0079 | 0.0020 |

## precision@10

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0040 | 0.0000 | 0.0030 | 0.0020 |
| full_bge_m3 | 0.0228 | 0.0079 | 0.0238 | 0.0188 |
| full_frida | 0.0248 | 0.0020 | 0.0188 | 0.0119 |
| full_e5_large_instruct | 0.0109 | 0.0030 | 0.0218 | 0.0109 |
| full_jina_embeddings_v3 | 0.0089 | 0.0020 | 0.0149 | 0.0059 |
| full_default_clean | 0.0040 | 0.0000 | 0.0040 | 0.0030 |

## mrr

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0144 | 0.0007 | 0.0116 | 0.0118 |
| full_bge_m3 | 0.0442 | 0.0143 | 0.0370 | 0.0299 |
| full_frida | 0.0402 | 0.0083 | 0.0354 | 0.0218 |
| full_e5_large_instruct | 0.0520 | 0.0062 | 0.0710 | 0.0487 |
| full_jina_embeddings_v3 | 0.0336 | 0.0116 | 0.0537 | 0.0173 |
| full_default_clean | 0.0143 | 0.0005 | 0.0127 | 0.0127 |

## ndcg@3

| scenario | dense_only | hyde | mqr | hyde_mqr_dense_bm25_rrf |
|---|---|---|---|---|
| full_mpnet | 0.0091 | 0.0000 | 0.0099 | 0.0061 |
| full_bge_m3 | 0.0319 | 0.0099 | 0.0291 | 0.0228 |
| full_frida | 0.0259 | 0.0038 | 0.0259 | 0.0121 |
| full_e5_large_instruct | 0.0513 | 0.0000 | 0.0677 | 0.0420 |
| full_jina_embeddings_v3 | 0.0297 | 0.0061 | 0.0487 | 0.0141 |
| full_default_clean | 0.0091 | 0.0000 | 0.0099 | 0.0061 |

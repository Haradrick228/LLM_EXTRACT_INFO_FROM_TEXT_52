# F2 hybrid retrieval sweep

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_015456_f2_hybrid_rrf_sweep`

Метрики по subsection — taken from `substring` hit_fn (primary), полный набор см. в `*.json`.

RRF k = `60`; fetch_k sweep = `[40, 100, 200, 400]`

## recall@5

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0099 | 0.0099 | 0.0149 |
| full_mpnet | 100 | 0.0099 | 0.0099 | 0.0149 |
| full_mpnet | 200 | 0.0099 | 0.0099 | 0.0347 |
| full_mpnet | 400 | 0.0099 | 0.0099 | 0.0347 |
| full_bge_m3 | 40 | 0.0347 | 0.0099 | 0.0248 |
| full_bge_m3 | 100 | 0.0347 | 0.0099 | 0.0248 |
| full_bge_m3 | 200 | 0.0347 | 0.0099 | 0.0248 |
| full_bge_m3 | 400 | 0.0347 | 0.0099 | 0.0248 |
| full_frida | 40 | 0.0594 | 0.0099 | 0.0198 |
| full_frida | 100 | 0.0594 | 0.0099 | 0.0198 |
| full_frida | 200 | 0.0594 | 0.0099 | 0.0198 |
| full_frida | 400 | 0.0594 | 0.0099 | 0.0198 |
| full_e5_large_instruct | 40 | 0.0594 | 0.0099 | 0.0644 |
| full_e5_large_instruct | 100 | 0.0594 | 0.0099 | 0.0644 |
| full_e5_large_instruct | 200 | 0.0644 | 0.0099 | 0.0644 |
| full_e5_large_instruct | 400 | 0.0644 | 0.0099 | 0.0743 |
| full_jina_embeddings_v3 | 40 | 0.0297 | 0.0099 | 0.0248 |
| full_jina_embeddings_v3 | 100 | 0.0297 | 0.0099 | 0.0248 |
| full_jina_embeddings_v3 | 200 | 0.0297 | 0.0099 | 0.0248 |
| full_jina_embeddings_v3 | 400 | 0.0297 | 0.0099 | 0.0248 |
| full_default_clean | 40 | 0.0099 | 0.0099 | 0.0149 |
| full_default_clean | 100 | 0.0099 | 0.0099 | 0.0149 |
| full_default_clean | 200 | 0.0099 | 0.0099 | 0.0248 |
| full_default_clean | 400 | 0.0099 | 0.0099 | 0.0248 |

## recall@10

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0198 | 0.0347 | 0.0149 |
| full_mpnet | 100 | 0.0198 | 0.0347 | 0.0149 |
| full_mpnet | 200 | 0.0099 | 0.0347 | 0.0347 |
| full_mpnet | 400 | 0.0099 | 0.0347 | 0.0347 |
| full_bge_m3 | 40 | 0.0446 | 0.0347 | 0.0347 |
| full_bge_m3 | 100 | 0.0446 | 0.0347 | 0.0347 |
| full_bge_m3 | 200 | 0.0446 | 0.0347 | 0.0347 |
| full_bge_m3 | 400 | 0.0446 | 0.0347 | 0.0347 |
| full_frida | 40 | 0.0644 | 0.0347 | 0.0594 |
| full_frida | 100 | 0.0644 | 0.0347 | 0.0594 |
| full_frida | 200 | 0.0644 | 0.0347 | 0.0594 |
| full_frida | 400 | 0.0644 | 0.0347 | 0.0594 |
| full_e5_large_instruct | 40 | 0.0594 | 0.0347 | 0.0743 |
| full_e5_large_instruct | 100 | 0.0594 | 0.0347 | 0.0743 |
| full_e5_large_instruct | 200 | 0.0644 | 0.0347 | 0.0743 |
| full_e5_large_instruct | 400 | 0.0644 | 0.0347 | 0.0842 |
| full_jina_embeddings_v3 | 40 | 0.0297 | 0.0347 | 0.0347 |
| full_jina_embeddings_v3 | 100 | 0.0297 | 0.0347 | 0.0347 |
| full_jina_embeddings_v3 | 200 | 0.0297 | 0.0347 | 0.0347 |
| full_jina_embeddings_v3 | 400 | 0.0297 | 0.0347 | 0.0347 |
| full_default_clean | 40 | 0.0198 | 0.0347 | 0.0149 |
| full_default_clean | 100 | 0.0198 | 0.0347 | 0.0149 |
| full_default_clean | 200 | 0.0099 | 0.0347 | 0.0248 |
| full_default_clean | 400 | 0.0099 | 0.0347 | 0.0248 |

## recall@20

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0198 | 0.0545 | 0.0446 |
| full_mpnet | 100 | 0.0198 | 0.0545 | 0.0446 |
| full_mpnet | 200 | 0.0198 | 0.0545 | 0.0545 |
| full_mpnet | 400 | 0.0198 | 0.0545 | 0.0545 |
| full_bge_m3 | 40 | 0.0446 | 0.0545 | 0.0545 |
| full_bge_m3 | 100 | 0.0446 | 0.0545 | 0.0545 |
| full_bge_m3 | 200 | 0.0446 | 0.0545 | 0.0545 |
| full_bge_m3 | 400 | 0.0446 | 0.0545 | 0.0545 |
| full_frida | 40 | 0.0743 | 0.0545 | 0.0644 |
| full_frida | 100 | 0.0743 | 0.0545 | 0.0644 |
| full_frida | 200 | 0.0743 | 0.0545 | 0.0644 |
| full_frida | 400 | 0.0644 | 0.0545 | 0.0644 |
| full_e5_large_instruct | 40 | 0.0594 | 0.0545 | 0.0743 |
| full_e5_large_instruct | 100 | 0.0594 | 0.0545 | 0.0743 |
| full_e5_large_instruct | 200 | 0.0644 | 0.0545 | 0.0743 |
| full_e5_large_instruct | 400 | 0.0644 | 0.0545 | 0.0842 |
| full_jina_embeddings_v3 | 40 | 0.0396 | 0.0545 | 0.0545 |
| full_jina_embeddings_v3 | 100 | 0.0396 | 0.0545 | 0.0545 |
| full_jina_embeddings_v3 | 200 | 0.0396 | 0.0545 | 0.0644 |
| full_jina_embeddings_v3 | 400 | 0.0396 | 0.0545 | 0.0644 |
| full_default_clean | 40 | 0.0198 | 0.0545 | 0.0446 |
| full_default_clean | 100 | 0.0198 | 0.0545 | 0.0446 |
| full_default_clean | 200 | 0.0198 | 0.0545 | 0.0446 |
| full_default_clean | 400 | 0.0198 | 0.0545 | 0.0446 |

## precision@5

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0059 | 0.0198 | 0.0139 |
| full_mpnet | 100 | 0.0059 | 0.0198 | 0.0139 |
| full_mpnet | 200 | 0.0059 | 0.0198 | 0.0158 |
| full_mpnet | 400 | 0.0059 | 0.0198 | 0.0158 |
| full_bge_m3 | 40 | 0.0198 | 0.0198 | 0.0198 |
| full_bge_m3 | 100 | 0.0198 | 0.0198 | 0.0198 |
| full_bge_m3 | 200 | 0.0198 | 0.0198 | 0.0218 |
| full_bge_m3 | 400 | 0.0198 | 0.0198 | 0.0218 |
| full_frida | 40 | 0.0238 | 0.0198 | 0.0198 |
| full_frida | 100 | 0.0238 | 0.0198 | 0.0198 |
| full_frida | 200 | 0.0238 | 0.0198 | 0.0198 |
| full_frida | 400 | 0.0238 | 0.0198 | 0.0198 |
| full_e5_large_instruct | 40 | 0.0198 | 0.0198 | 0.0297 |
| full_e5_large_instruct | 100 | 0.0198 | 0.0198 | 0.0317 |
| full_e5_large_instruct | 200 | 0.0218 | 0.0198 | 0.0356 |
| full_e5_large_instruct | 400 | 0.0257 | 0.0198 | 0.0416 |
| full_jina_embeddings_v3 | 40 | 0.0158 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 100 | 0.0158 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 200 | 0.0158 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 400 | 0.0158 | 0.0198 | 0.0198 |
| full_default_clean | 40 | 0.0059 | 0.0198 | 0.0139 |
| full_default_clean | 100 | 0.0059 | 0.0198 | 0.0139 |
| full_default_clean | 200 | 0.0059 | 0.0198 | 0.0139 |
| full_default_clean | 400 | 0.0059 | 0.0198 | 0.0139 |

## precision@10

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0040 | 0.0257 | 0.0129 |
| full_mpnet | 100 | 0.0040 | 0.0257 | 0.0129 |
| full_mpnet | 200 | 0.0030 | 0.0257 | 0.0139 |
| full_mpnet | 400 | 0.0030 | 0.0257 | 0.0139 |
| full_bge_m3 | 40 | 0.0228 | 0.0257 | 0.0198 |
| full_bge_m3 | 100 | 0.0228 | 0.0257 | 0.0188 |
| full_bge_m3 | 200 | 0.0228 | 0.0257 | 0.0188 |
| full_bge_m3 | 400 | 0.0228 | 0.0257 | 0.0188 |
| full_frida | 40 | 0.0248 | 0.0257 | 0.0218 |
| full_frida | 100 | 0.0248 | 0.0257 | 0.0218 |
| full_frida | 200 | 0.0248 | 0.0257 | 0.0218 |
| full_frida | 400 | 0.0248 | 0.0257 | 0.0208 |
| full_e5_large_instruct | 40 | 0.0109 | 0.0257 | 0.0218 |
| full_e5_large_instruct | 100 | 0.0109 | 0.0257 | 0.0228 |
| full_e5_large_instruct | 200 | 0.0129 | 0.0257 | 0.0267 |
| full_e5_large_instruct | 400 | 0.0188 | 0.0257 | 0.0297 |
| full_jina_embeddings_v3 | 40 | 0.0089 | 0.0257 | 0.0178 |
| full_jina_embeddings_v3 | 100 | 0.0089 | 0.0257 | 0.0178 |
| full_jina_embeddings_v3 | 200 | 0.0089 | 0.0257 | 0.0178 |
| full_jina_embeddings_v3 | 400 | 0.0089 | 0.0257 | 0.0178 |
| full_default_clean | 40 | 0.0040 | 0.0257 | 0.0129 |
| full_default_clean | 100 | 0.0040 | 0.0257 | 0.0129 |
| full_default_clean | 200 | 0.0030 | 0.0257 | 0.0129 |
| full_default_clean | 400 | 0.0030 | 0.0257 | 0.0129 |

## mrr

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0142 | 0.0252 | 0.0191 |
| full_mpnet | 100 | 0.0144 | 0.0252 | 0.0193 |
| full_mpnet | 200 | 0.0144 | 0.0252 | 0.0267 |
| full_mpnet | 400 | 0.0145 | 0.0252 | 0.0257 |
| full_bge_m3 | 40 | 0.0442 | 0.0252 | 0.0422 |
| full_bge_m3 | 100 | 0.0442 | 0.0252 | 0.0374 |
| full_bge_m3 | 200 | 0.0442 | 0.0252 | 0.0374 |
| full_bge_m3 | 400 | 0.0442 | 0.0252 | 0.0374 |
| full_frida | 40 | 0.0401 | 0.0252 | 0.0355 |
| full_frida | 100 | 0.0402 | 0.0252 | 0.0354 |
| full_frida | 200 | 0.0401 | 0.0252 | 0.0354 |
| full_frida | 400 | 0.0401 | 0.0252 | 0.0304 |
| full_e5_large_instruct | 40 | 0.0520 | 0.0252 | 0.0688 |
| full_e5_large_instruct | 100 | 0.0520 | 0.0252 | 0.0689 |
| full_e5_large_instruct | 200 | 0.0569 | 0.0252 | 0.0689 |
| full_e5_large_instruct | 400 | 0.0570 | 0.0252 | 0.0839 |
| full_jina_embeddings_v3 | 40 | 0.0336 | 0.0252 | 0.0374 |
| full_jina_embeddings_v3 | 100 | 0.0336 | 0.0252 | 0.0376 |
| full_jina_embeddings_v3 | 200 | 0.0336 | 0.0252 | 0.0385 |
| full_jina_embeddings_v3 | 400 | 0.0340 | 0.0252 | 0.0385 |
| full_default_clean | 40 | 0.0142 | 0.0252 | 0.0191 |
| full_default_clean | 100 | 0.0143 | 0.0252 | 0.0193 |
| full_default_clean | 200 | 0.0143 | 0.0252 | 0.0242 |
| full_default_clean | 400 | 0.0145 | 0.0252 | 0.0233 |

## ndcg@3

| scenario | fetch_k | dense_only | bm25_only | hybrid_rrf |
|---|---|---|---|---|
| full_mpnet | 40 | 0.0091 | 0.0198 | 0.0137 |
| full_mpnet | 100 | 0.0091 | 0.0198 | 0.0137 |
| full_mpnet | 200 | 0.0091 | 0.0198 | 0.0161 |
| full_mpnet | 400 | 0.0091 | 0.0198 | 0.0149 |
| full_bge_m3 | 40 | 0.0319 | 0.0198 | 0.0388 |
| full_bge_m3 | 100 | 0.0319 | 0.0198 | 0.0358 |
| full_bge_m3 | 200 | 0.0319 | 0.0198 | 0.0358 |
| full_bge_m3 | 400 | 0.0319 | 0.0198 | 0.0358 |
| full_frida | 40 | 0.0259 | 0.0198 | 0.0297 |
| full_frida | 100 | 0.0259 | 0.0198 | 0.0297 |
| full_frida | 200 | 0.0259 | 0.0198 | 0.0297 |
| full_frida | 400 | 0.0259 | 0.0198 | 0.0267 |
| full_e5_large_instruct | 40 | 0.0513 | 0.0198 | 0.0674 |
| full_e5_large_instruct | 100 | 0.0513 | 0.0198 | 0.0674 |
| full_e5_large_instruct | 200 | 0.0551 | 0.0198 | 0.0704 |
| full_e5_large_instruct | 400 | 0.0551 | 0.0198 | 0.0826 |
| full_jina_embeddings_v3 | 40 | 0.0297 | 0.0198 | 0.0327 |
| full_jina_embeddings_v3 | 100 | 0.0297 | 0.0198 | 0.0327 |
| full_jina_embeddings_v3 | 200 | 0.0297 | 0.0198 | 0.0319 |
| full_jina_embeddings_v3 | 400 | 0.0297 | 0.0198 | 0.0319 |
| full_default_clean | 40 | 0.0091 | 0.0198 | 0.0137 |
| full_default_clean | 100 | 0.0091 | 0.0198 | 0.0137 |
| full_default_clean | 200 | 0.0091 | 0.0198 | 0.0161 |
| full_default_clean | 400 | 0.0091 | 0.0198 | 0.0149 |

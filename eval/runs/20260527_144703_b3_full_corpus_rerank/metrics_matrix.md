# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260527_144703_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт). При `--metric-modes substring,token_f1,semantic` появляются ключи вида `token_f1::recall@10`.

## recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0050 | 0.0198 | 0.0198 |
| full_bge_m3 | 0.0248 | 0.0545 | 0.0446 |
| full_frida | 0.0396 | 0.0644 | 0.0644 |
| full_e5_large_instruct | 0.0050 | 0.0594 | 0.0594 |
| full_jina_embeddings_v3 | 0.0149 | 0.0594 | 0.0297 |
| full_default_clean | 0.0050 | 0.0198 | 0.0198 |

## recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0099 | 0.0198 | 0.0198 |
| full_bge_m3 | 0.0248 | 0.0545 | 0.0446 |
| full_frida | 0.0545 | 0.0743 | 0.0743 |
| full_e5_large_instruct | 0.0297 | 0.0693 | 0.0594 |
| full_jina_embeddings_v3 | 0.0198 | 0.0594 | 0.0396 |
| full_default_clean | 0.0099 | 0.0198 | 0.0198 |

## recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0198 | 0.0099 |
| full_bge_m3 | 0.0099 | 0.0248 | 0.0347 |
| full_frida | 0.0000 | 0.0297 | 0.0594 |
| full_e5_large_instruct | 0.0050 | 0.0297 | 0.0594 |
| full_jina_embeddings_v3 | 0.0000 | 0.0297 | 0.0297 |
| full_default_clean | 0.0000 | 0.0198 | 0.0099 |

## mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0025 | 0.0297 | 0.0142 |
| full_bge_m3 | 0.0086 | 0.0209 | 0.0442 |
| full_frida | 0.0072 | 0.0236 | 0.0401 |
| full_e5_large_instruct | 0.0132 | 0.0224 | 0.0520 |
| full_jina_embeddings_v3 | 0.0041 | 0.0208 | 0.0336 |
| full_default_clean | 0.0025 | 0.0297 | 0.0142 |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0297 | 0.0091 |
| full_bge_m3 | 0.0000 | 0.0061 | 0.0319 |
| full_frida | 0.0000 | 0.0061 | 0.0259 |
| full_e5_large_instruct | 0.0061 | 0.0061 | 0.0513 |
| full_jina_embeddings_v3 | 0.0000 | 0.0061 | 0.0297 |
| full_default_clean | 0.0000 | 0.0297 | 0.0091 |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0030 | 0.0109 | 0.0040 |
| full_bge_m3 | 0.0109 | 0.0158 | 0.0228 |
| full_frida | 0.0059 | 0.0218 | 0.0248 |
| full_e5_large_instruct | 0.0050 | 0.0188 | 0.0109 |
| full_jina_embeddings_v3 | 0.0030 | 0.0198 | 0.0089 |
| full_default_clean | 0.0030 | 0.0109 | 0.0040 |

## precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0030 | 0.0099 | 0.0054 |
| full_bge_m3 | 0.0124 | 0.0208 | 0.0178 |
| full_frida | 0.0109 | 0.0272 | 0.0238 |
| full_e5_large_instruct | 0.0084 | 0.0203 | 0.0149 |
| full_jina_embeddings_v3 | 0.0054 | 0.0168 | 0.0069 |
| full_default_clean | 0.0030 | 0.0099 | 0.0054 |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0178 | 0.0059 |
| full_bge_m3 | 0.0059 | 0.0099 | 0.0198 |
| full_frida | 0.0000 | 0.0099 | 0.0238 |
| full_e5_large_instruct | 0.0040 | 0.0139 | 0.0198 |
| full_jina_embeddings_v3 | 0.0000 | 0.0119 | 0.0158 |
| full_default_clean | 0.0000 | 0.0178 | 0.0059 |

## semantic::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.3830 | 0.5692 | 0.4920 |
| full_bge_m3 | 0.3725 | 0.6079 | 0.5281 |
| full_frida | 0.3402 | 0.5278 | 0.4814 |
| full_e5_large_instruct | 0.3428 | 0.5831 | 0.4587 |
| full_jina_embeddings_v3 | 0.4289 | 0.6157 | 0.5285 |
| full_default_clean | 0.3784 | 0.5668 | 0.5029 |

## semantic::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.3763 | 0.5639 | 0.4886 |
| full_bge_m3 | 0.3795 | 0.6037 | 0.5273 |
| full_frida | 0.3270 | 0.5392 | 0.4966 |
| full_e5_large_instruct | 0.3395 | 0.6035 | 0.4803 |
| full_jina_embeddings_v3 | 0.4389 | 0.6012 | 0.5491 |
| full_default_clean | 0.3677 | 0.5540 | 0.5034 |

## semantic::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.3376 | 0.4178 | 0.3752 |
| full_bge_m3 | 0.3089 | 0.4020 | 0.3604 |
| full_frida | 0.2604 | 0.3446 | 0.3079 |
| full_e5_large_instruct | 0.2634 | 0.3693 | 0.3020 |
| full_jina_embeddings_v3 | 0.3267 | 0.4089 | 0.3762 |
| full_default_clean | 0.3376 | 0.4149 | 0.3772 |

## semantic::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.3391 | 0.3748 | 0.3619 |
| full_bge_m3 | 0.3089 | 0.3441 | 0.3322 |
| full_frida | 0.2535 | 0.2965 | 0.2748 |
| full_e5_large_instruct | 0.2688 | 0.3203 | 0.2777 |
| full_jina_embeddings_v3 | 0.3203 | 0.3609 | 0.3485 |
| full_default_clean | 0.3396 | 0.3762 | 0.3609 |

## semantic::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.3129 | 0.4475 | 0.3980 |
| full_bge_m3 | 0.2990 | 0.4733 | 0.3921 |
| full_frida | 0.2475 | 0.4040 | 0.3525 |
| full_e5_large_instruct | 0.2495 | 0.4455 | 0.3307 |
| full_jina_embeddings_v3 | 0.3327 | 0.4752 | 0.4059 |
| full_default_clean | 0.3129 | 0.4436 | 0.4040 |

## semantic::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.5792 | 0.6634 | 0.6089 |
| full_bge_m3 | 0.6287 | 0.6733 | 0.6188 |
| full_frida | 0.5594 | 0.6040 | 0.6238 |
| full_e5_large_instruct | 0.5743 | 0.6634 | 0.6040 |
| full_jina_embeddings_v3 | 0.6139 | 0.7129 | 0.6881 |
| full_default_clean | 0.5891 | 0.6535 | 0.6188 |

## semantic::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.6436 | 0.7178 | 0.6782 |
| full_bge_m3 | 0.6832 | 0.7129 | 0.6782 |
| full_frida | 0.6188 | 0.6535 | 0.6634 |
| full_e5_large_instruct | 0.6782 | 0.7079 | 0.6832 |
| full_jina_embeddings_v3 | 0.6931 | 0.7673 | 0.7277 |
| full_default_clean | 0.6634 | 0.7178 | 0.6881 |

## semantic::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.4901 | 0.6040 | 0.5644 |
| full_bge_m3 | 0.5000 | 0.6337 | 0.5693 |
| full_frida | 0.4604 | 0.5594 | 0.5693 |
| full_e5_large_instruct | 0.4505 | 0.6287 | 0.5396 |
| full_jina_embeddings_v3 | 0.5446 | 0.6634 | 0.6386 |
| full_default_clean | 0.4901 | 0.6040 | 0.5743 |

## substring::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0025 | 0.0297 | 0.0142 |
| full_bge_m3 | 0.0086 | 0.0209 | 0.0442 |
| full_frida | 0.0072 | 0.0236 | 0.0401 |
| full_e5_large_instruct | 0.0132 | 0.0224 | 0.0520 |
| full_jina_embeddings_v3 | 0.0041 | 0.0208 | 0.0336 |
| full_default_clean | 0.0025 | 0.0297 | 0.0142 |

## substring::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0297 | 0.0091 |
| full_bge_m3 | 0.0000 | 0.0061 | 0.0319 |
| full_frida | 0.0000 | 0.0061 | 0.0259 |
| full_e5_large_instruct | 0.0061 | 0.0061 | 0.0513 |
| full_jina_embeddings_v3 | 0.0000 | 0.0061 | 0.0297 |
| full_default_clean | 0.0000 | 0.0297 | 0.0091 |

## substring::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0030 | 0.0109 | 0.0040 |
| full_bge_m3 | 0.0109 | 0.0158 | 0.0228 |
| full_frida | 0.0059 | 0.0218 | 0.0248 |
| full_e5_large_instruct | 0.0050 | 0.0188 | 0.0109 |
| full_jina_embeddings_v3 | 0.0030 | 0.0198 | 0.0089 |
| full_default_clean | 0.0030 | 0.0109 | 0.0040 |

## substring::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0030 | 0.0099 | 0.0054 |
| full_bge_m3 | 0.0124 | 0.0208 | 0.0178 |
| full_frida | 0.0109 | 0.0272 | 0.0238 |
| full_e5_large_instruct | 0.0084 | 0.0203 | 0.0149 |
| full_jina_embeddings_v3 | 0.0054 | 0.0168 | 0.0069 |
| full_default_clean | 0.0030 | 0.0099 | 0.0054 |

## substring::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0178 | 0.0059 |
| full_bge_m3 | 0.0059 | 0.0099 | 0.0198 |
| full_frida | 0.0000 | 0.0099 | 0.0238 |
| full_e5_large_instruct | 0.0040 | 0.0139 | 0.0198 |
| full_jina_embeddings_v3 | 0.0000 | 0.0119 | 0.0158 |
| full_default_clean | 0.0000 | 0.0178 | 0.0059 |

## substring::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0050 | 0.0198 | 0.0198 |
| full_bge_m3 | 0.0248 | 0.0545 | 0.0446 |
| full_frida | 0.0396 | 0.0644 | 0.0644 |
| full_e5_large_instruct | 0.0050 | 0.0594 | 0.0594 |
| full_jina_embeddings_v3 | 0.0149 | 0.0594 | 0.0297 |
| full_default_clean | 0.0050 | 0.0198 | 0.0198 |

## substring::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0099 | 0.0198 | 0.0198 |
| full_bge_m3 | 0.0248 | 0.0545 | 0.0446 |
| full_frida | 0.0545 | 0.0743 | 0.0743 |
| full_e5_large_instruct | 0.0297 | 0.0693 | 0.0594 |
| full_jina_embeddings_v3 | 0.0198 | 0.0594 | 0.0396 |
| full_default_clean | 0.0099 | 0.0198 | 0.0198 |

## substring::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0198 | 0.0099 |
| full_bge_m3 | 0.0099 | 0.0248 | 0.0347 |
| full_frida | 0.0000 | 0.0297 | 0.0594 |
| full_e5_large_instruct | 0.0050 | 0.0297 | 0.0594 |
| full_jina_embeddings_v3 | 0.0000 | 0.0297 | 0.0297 |
| full_default_clean | 0.0000 | 0.0198 | 0.0099 |

## token_f1::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0036 | 0.0484 | 0.0185 |
| full_frida | 0.0026 | 0.0113 | 0.0032 |
| full_e5_large_instruct | 0.0021 | 0.0124 | 0.0198 |
| full_jina_embeddings_v3 | 0.0022 | 0.0149 | 0.0114 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0000 | 0.0495 | 0.0099 |
| full_frida | 0.0000 | 0.0099 | 0.0000 |
| full_e5_large_instruct | 0.0000 | 0.0099 | 0.0198 |
| full_jina_embeddings_v3 | 0.0000 | 0.0099 | 0.0099 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0010 | 0.0158 | 0.0069 |
| full_frida | 0.0020 | 0.0030 | 0.0010 |
| full_e5_large_instruct | 0.0010 | 0.0030 | 0.0030 |
| full_jina_embeddings_v3 | 0.0000 | 0.0040 | 0.0020 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0020 | 0.0079 | 0.0079 |
| full_frida | 0.0015 | 0.0015 | 0.0015 |
| full_e5_large_instruct | 0.0015 | 0.0015 | 0.0015 |
| full_jina_embeddings_v3 | 0.0015 | 0.0020 | 0.0020 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0000 | 0.0297 | 0.0079 |
| full_frida | 0.0020 | 0.0040 | 0.0020 |
| full_e5_large_instruct | 0.0000 | 0.0059 | 0.0059 |
| full_jina_embeddings_v3 | 0.0000 | 0.0079 | 0.0040 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0099 | 0.0693 | 0.0396 |
| full_frida | 0.0099 | 0.0198 | 0.0099 |
| full_e5_large_instruct | 0.0099 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 0.0000 | 0.0297 | 0.0099 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0297 | 0.0693 | 0.0693 |
| full_frida | 0.0198 | 0.0198 | 0.0198 |
| full_e5_large_instruct | 0.0198 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 0.0297 | 0.0297 | 0.0297 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

## token_f1::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet | 0.0000 | 0.0000 | 0.0000 |
| full_bge_m3 | 0.0000 | 0.0594 | 0.0297 |
| full_frida | 0.0099 | 0.0099 | 0.0099 |
| full_e5_large_instruct | 0.0000 | 0.0198 | 0.0198 |
| full_jina_embeddings_v3 | 0.0000 | 0.0297 | 0.0099 |
| full_default_clean | 0.0000 | 0.0000 | 0.0000 |

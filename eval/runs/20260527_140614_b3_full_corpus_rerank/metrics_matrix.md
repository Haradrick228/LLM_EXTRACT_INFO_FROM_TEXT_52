# B3 retrieval metrics matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260527_140614_b3_full_corpus_rerank`

Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт). При `--metric-modes substring,token_f1,semantic` появляются ключи вида `token_f1::recall@10`.

## recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0396 | 0.0644 | 0.0644 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0545 | 0.0743 | 0.0743 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0594 | 0.0594 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0072 | 0.0401 | 0.0401 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0259 | 0.0259 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0059 | 0.0248 | 0.0248 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0109 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0072 | 0.0401 | 0.0401 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0259 | 0.0259 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0059 | 0.0248 | 0.0248 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0109 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0396 | 0.0644 | 0.0644 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0545 | 0.0743 | 0.0743 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## semantic::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0594 | 0.0594 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0072 | 0.0401 | 0.0401 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0259 | 0.0259 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0059 | 0.0248 | 0.0248 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0109 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0238 | 0.0238 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0396 | 0.0644 | 0.0644 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0545 | 0.0743 | 0.0743 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## substring::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0594 | 0.0594 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::mrr

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0026 | 0.0032 | 0.0032 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::ndcg@3

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0000 | 0.0000 | 0.0000 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::precision@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0020 | 0.0010 | 0.0010 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::precision@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0015 | 0.0015 | 0.0015 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::precision@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0020 | 0.0020 | 0.0020 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::recall@10

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0099 | 0.0099 | 0.0099 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::recall@20

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0198 | 0.0198 | 0.0198 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

## token_f1::recall@5

| scenario | baseline_mpnet_rerank | rerank_bge_v2_m3 | rerank_jina_v3 |
|---|---|---|---|
| full_mpnet |  |  |  |
| full_bge_m3 |  |  |  |
| full_frida | 0.0099 | 0.0099 | 0.0099 |
| full_e5_large_instruct |  |  |  |
| full_jina_embeddings_v3 |  |  |  |
| full_default_clean |  |  |  |

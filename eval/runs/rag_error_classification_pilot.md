# RAG error classification (pilot)

- Trace: `C:/IT/VS Code/LLM_EXTRACT_INFO_FROM_TEXT_52_DEV/eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/trace.jsonl`
- Answer-eval: `C:/IT/VS Code/LLM_EXTRACT_INFO_FROM_TEXT_52_DEV/eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/answer_eval_results.jsonl`
- n_queries: **101**
- Rules: type1: recall@k=0 or ladder A/B; type2: recall@k>0 and ref_sim<threshold; type0_ok: recall@k>0 and (no answer-eval or ref_sim>=threshold).

## Counts

| Class | Count | Fraction |
|-------|------:|---------:|
| `type0_ok` | 5 | 5.0% |
| `type1_retrieval` | 95 | 94.1% |
| `type2_generation` | 1 | 1.0% |

## Examples: `type1_retrieval`

- idx=0 R@10=0.0 ref_sim=0.8173096776008606 — Что такое серия образовательных материалов «Школа возможностей»?
- idx=1 R@10=0.0 ref_sim=0.5828412771224976 — Для чего создавалась программа «Развивающая среда»?
- idx=2 R@10=0.0 ref_sim=0.6128891706466675 — Какова миссия фонда «Вклад в будущее» в рамках программы «Развивающая среда»?
- idx=4 R@10=0.0 ref_sim=0.707442045211792 — Сколько произведений включает Хрестоматия?
- idx=5 R@10=0.0 ref_sim=0.9252535700798035 — Сколько произведений имеют задания и игры?

## Examples: `type2_generation`

- idx=70 R@10=0.5 ref_sim=0.4719766080379486 — Какую поддержку предлагает Хрестоматия для всех возрастов?

## Examples: `type0_ok`

- idx=3 R@10=1.0 ref_sim=0.6848108172416687 — Какие возрастные группы охватывает Хрестоматия?
- idx=12 R@10=0.5 ref_sim=0.6555535793304443 — Что такое раздел «Хрестоматия 14–18 лет»?
- idx=18 R@10=1.0 ref_sim=0.5837444067001343 — Для кого предназначены материалы возрастной группы 11–13 лет?
- idx=19 R@10=1.0 ref_sim=0.576202392578125 — Для кого предназначены материалы возрастной группы 14–18 лет?
- idx=58 R@10=0.5 ref_sim=0.7483652830123901 — Какую информацию можно найти в разделе 14–18 лет?

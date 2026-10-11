"""CLI и оркестраторы оценки retrieval/rerank/answer.

Все скрипты запускаются как ``python -m eval.retrieval.<module>``:

- :mod:`eval.retrieval.run_retrieval_eval`       — одиночный прогон
  (одна коллекция × один реранкер) на gold-датасете;
- :mod:`eval.retrieval.run_retrieval_matrix`     — матрица сценариев на
  полном корпусе (бывший ``run_full_corpus_rerank_matrix``);
- :mod:`eval.retrieval.run_profile_metrics`      — матрица по
  ``eval/config/rag_profiles.yaml``;
- :mod:`eval.retrieval.run_eval_with_profile`    — разовый прогон под профилем;
- :mod:`eval.retrieval.run_answer_quality_eval`  — оценка качества ответов LLM
  (бывший ``run_answer_eval``).
"""

"""Gold/eval-датасеты и инструменты их сборки.

CLI:

- :mod:`eval.datasets.materialize_gold_v2_from_answer_ref` — собрать gold v2
  из ``eval_set_with_answer_ref.jsonl`` (основной путь, без LLM);
- :mod:`eval.datasets.build_gold_v2_from_chroma_chunks`    — собрать gold v2
  из живой Chroma + LLM-вопросы;
- :mod:`eval.datasets.generate_synthetic_gold_v2`          — синтетика для CI;
- :mod:`eval.datasets.validate_gold_v2_schema`             — валидация JSONL.

Загрузчики и сами JSONL-датасеты также лежат в этом пакете
(``eval_set_gold_v2.jsonl``, ``chunks_sample.jsonl``).
"""

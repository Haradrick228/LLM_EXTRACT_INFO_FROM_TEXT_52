"""YAML-конфиги retrieval-стека.

Файлы:

- ``embedding_bases_fixture.yaml``       — B, 6 моделей на gold-фикстуре;
- ``embedding_bases_full_corpus.yaml``   — B, 6 моделей на полном корпусе;
- ``embedding_bases_chunking_variants.yaml`` — F5: parent-child / semantic;
- ``rag_profiles.yaml``                   — профили реранкеров и LLM;
- ``experiments_registry.yaml``           — реестр replay-экспериментов;
- ``prompts/hyde_hypothetical_document.txt`` — промпт HyDE (F3);
- ``prompts/multi_query_rewrite_variants.txt`` — промпт MQR (F3).
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

EMBEDDING_BASES_FIXTURE = CONFIG_DIR / "embedding_bases_fixture.yaml"
EMBEDDING_BASES_FULL_CORPUS = CONFIG_DIR / "embedding_bases_full_corpus.yaml"
EMBEDDING_BASES_CHUNKING_VARIANTS = CONFIG_DIR / "embedding_bases_chunking_variants.yaml"
RAG_PROFILES = CONFIG_DIR / "rag_profiles.yaml"
EXPERIMENTS_REGISTRY = CONFIG_DIR / "experiments_registry.yaml"
PROMPTS_DIR = CONFIG_DIR / "prompts"

__all__ = [
    "CONFIG_DIR",
    "EMBEDDING_BASES_FIXTURE",
    "EMBEDDING_BASES_FULL_CORPUS",
    "EMBEDDING_BASES_CHUNKING_VARIANTS",
    "RAG_PROFILES",
    "EXPERIMENTS_REGISTRY",
    "PROMPTS_DIR",
]

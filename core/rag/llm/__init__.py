"""LLM-клиенты RAG-слоя.

- :mod:`core.rag.llm.deepseek_llm_client` — клиент DeepSeek API.
Клиент Ollama и фабрика провайдеров живут в :mod:`core.graphrag.llm_provider`
(там же ``build_llm_client``); их в этот пакет переносить пока не нужно —
они и сейчас работают.
"""

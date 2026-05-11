"""LLM clients for RAG: DeepSeek API (default) or local Ollama."""
from __future__ import annotations

import os
import requests
from typing import Any, List, Dict, Optional, Union


def _ollama_resolved_model(requested: Optional[str]) -> str:
    if requested and requested.strip() and requested.strip() != "deepseek-chat":
        return requested.strip()
    return (os.getenv("OLLAMA_MODEL") or "qwen2.5:7b-instruct").strip()


class DeepSeekLLMClient:
    """Chat completions against DeepSeek API (requires DEEPSEEK_API_KEY)."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/chat/completions"
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.2,
            "max_tokens": max_tokens if max_tokens is not None else 1000,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)


class OllamaLLMClient:
    """Local Ollama HTTP API (/api/chat). No API key."""

    def __init__(self, model: Optional[str] = None):
        self.base = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model = _ollama_resolved_model(model)

    def _chat_url(self) -> str:
        return f"{self.base}/api/chat"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        opts: Dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = float(temperature)
        if max_tokens is not None:
            opts["num_predict"] = int(max_tokens)
        if opts:
            payload["options"] = opts

        response = requests.post(self._chat_url(), json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        msg = data.get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content and isinstance(data.get("response"), str):
            content = (data.get("response") or "").strip()
        return content

    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)


def build_llm_client(model: Optional[str] = None) -> Union[DeepSeekLLMClient, OllamaLLMClient]:
    """
    Factory: LLM_PROVIDER=ollama -> OllamaLLMClient; else DeepSeekLLMClient.
    Model resolution: explicit arg > LLM_PROVIDER_MODEL / LLM_MODEL env.
    """
    provider = (os.getenv("LLM_PROVIDER") or "deepseek").strip().lower()
    if provider in ("ollama", "local", "local_ollama"):
        return OllamaLLMClient(model=model or os.getenv("LLM_PROVIDER_MODEL") or os.getenv("LLM_MODEL"))
    return DeepSeekLLMClient(model=model or os.getenv("LLM_PROVIDER_MODEL") or os.getenv("LLM_MODEL"))


# Backward compatibility: existing imports `from core.graphrag.llm_provider import LLMClient`
LLMClient = DeepSeekLLMClient

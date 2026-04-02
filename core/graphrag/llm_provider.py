"""LLM Client for GraphRAG."""
import os
import requests
from typing import List, Dict, Any, Optional


class LLMClient:
    """Simple LLM client wrapper for DeepSeek API."""

    def __init__(self, model=None):
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/chat/completions"

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Send a chat message to DeepSeek API and return the response."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.2,
            "max_tokens": max_tokens if max_tokens is not None else 1000
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Send a prompt to DeepSeek API and return the response (alias for chat)."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

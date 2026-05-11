import os
from unittest.mock import MagicMock, patch

from core.graphrag.llm_provider import OllamaLLMClient, build_llm_client


def test_build_llm_ollama_branch():
    base = dict(os.environ)
    try:
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ.pop("DEEPSEEK_API_KEY", None)
        c = build_llm_client("qwen2.5:7b-instruct")
        assert isinstance(c, OllamaLLMClient)
        assert "qwen" in c.model.lower()
    finally:
        os.environ.clear()
        os.environ.update(base)


@patch("core.graphrag.llm_provider.requests.post")
def test_ollama_chat_parses_message(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"message": {"role": "assistant", "content": "  ok  "}}
    mock_post.return_value = mock_resp

    c = OllamaLLMClient(model="qwen2.5:7b-instruct")
    out = c.chat([{"role": "user", "content": "ping"}])
    assert out == "ok"
    mock_post.assert_called_once()

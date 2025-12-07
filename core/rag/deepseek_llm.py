from typing import Optional, List
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import Field

from langchain_core.outputs import ChatGeneration, ChatResult


class DeepSeekLLM(BaseChatModel):
    api_key: str = Field(..., description="API key for DeepSeek")
    base_url: str = Field(default="https://api.deepseek.com/v1")
    model_name: str = Field(default="deepseek-chat")
    temperature: float = Field(default=0.7)


    def _call_api(self, messages: List[BaseMessage]) -> str:

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": self._get_role(msg), "content": msg.content}
                for msg in messages
            ],
            "temperature": self.temperature
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()



        return response.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _get_role(msg: BaseMessage) -> str:
        if isinstance(msg, AIMessage):
            return "assistant"
        elif isinstance(msg, HumanMessage):
            return "user"
        elif hasattr(msg, "role"):
            return msg.role
        return "user"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            config: Optional[RunnableConfig] = None,
    ) -> ChatResult:
        content = self._call_api(messages)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "deepseek-custom"

import asyncio
from common.interfaces.user_interaction import BotAdapter

class RestAdapter(BotAdapter):
    """
    Адаптер для REST API, использующий RagService для обработки сообщений.
    """
    def __init__(self, rag_service):
        self.rag_service = rag_service

    async def handle_message(self, text: str) -> str:
        """
        Обрабатывает входящий текст через RagService и возвращает ответ.
        """
        loop = asyncio.get_event_loop()
        answer_text, _ = await loop.run_in_executor(
            None,
            lambda: self.rag_service.answer(text)
        )
        return answer_text

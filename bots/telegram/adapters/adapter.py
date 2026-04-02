from aiogram import Bot
from aiogram.types import FSInputFile

from common.interfaces.user_interaction import UserInteraction


class TelegramAdapter(UserInteraction):
    def __init__(self, bot: Bot):
        self.bot = bot

    async def send_message(self, user_id: str, text: str, **kwargs):
        await self.bot.send_message(chat_id=user_id, text=text, **kwargs)

    async def send_file(self, user_id: str, file_path: str, **kwargs):
        await self.bot.send_document(chat_id=user_id, document=FSInputFile(file_path), **kwargs)

    async def download_file(self, file_id: str, destination: str) -> str:
        """
        Скачивает файл из Telegram и сохраняет по пути destination.
        Возвращает путь к сохранённому файлу.
        """
        file = await self.bot.get_file(file_id)
        await self.bot.download_file(file.file_path, destination)
        return destination
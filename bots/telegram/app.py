"""Точка входа Telegram-бота и регистрация обработчиков."""
import asyncio
from contextlib import suppress
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

from common.config import Settings
from core.db_service import DBService
from core.rag.rag_service_wrapper import RagServiceWrapper
from core.services.file_library import FileLibraryService
from core.services.link_library_service import LinkLibraryService
from core.services.link_service import LinkService
from core.services.upload_service import UploadService

from bots.telegram.adapters.adapter import TelegramAdapter
from bots.telegram.handlers.back import BackHandler
from bots.telegram.handlers.feedback import FeedbackHandler
from bots.telegram.handlers.library import LibraryHandler
from bots.telegram.handlers.link import LinkHandler
from bots.telegram.handlers.link_library import LinkLibraryHandler
from bots.telegram.handlers.nps import NPSHandler
from bots.telegram.handlers.question import QuestionHandler
from bots.telegram.handlers.start import StartHandler
from bots.telegram.handlers.upload import UploadHandler
from bots.rest_api.rest_main import start_rest_api

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

settings = Settings()

bot = Bot(token=settings.bot_token)
storage = MemoryStorage()
dispatcher = Dispatcher(storage=storage)
adapter = TelegramAdapter(bot)

db = DBService()
rag_service = RagServiceWrapper(settings, db)


def setup_handlers() -> None:
    """Регистрирует все обработчики и роутеры в диспетчере."""

    file_library = FileLibraryService(
        list_files_fn=rag_service.list_files,
        scan_dirs=[settings.folder_path] if settings.folder_path else [],
        page_size=30,
    )
    link_service = LinkService(rag_service, db)
    link_library = LinkLibraryService(link_service.list_links, page_size=10)
    upload_service = UploadService(rag_service, settings.folder_path)

    handlers = [
        BackHandler(adapter, settings),
        StartHandler(adapter, settings),
        LibraryHandler(adapter, file_library, rag_service),
        LinkHandler(adapter, link_service, settings),
        LinkLibraryHandler(adapter, link_library),
        UploadHandler(adapter, upload_service, settings),
        QuestionHandler(adapter, rag_service, db, settings),
        FeedbackHandler(adapter, db),
        NPSHandler(adapter, db),
    ]

    for handler in handlers:
        router = getattr(handler, "router", None)
        if router is not None:
            dispatcher.include_router(router)

    link_library_handler = next(
        handler for handler in handlers if isinstance(handler, LinkLibraryHandler)
    )
    dispatcher.message.register(link_library_handler.show_page, Command(commands=["links"]))
    dispatcher.callback_query.register(
        link_library_handler.on_page_callback,
        F.data.startswith("page:"),
    )
async def main() -> None:
    """Запускает Telegram-бота и связанный REST API."""

    setup_handlers()

    rest_task = asyncio.create_task(
        asyncio.to_thread(start_rest_api, rag_service),
        name="rest-api",
    )
    try:
        await dispatcher.start_polling(bot)
    finally:
        rest_task.cancel()
        with suppress(asyncio.CancelledError):
            await rest_task


if __name__ == "__main__":
    asyncio.run(main())


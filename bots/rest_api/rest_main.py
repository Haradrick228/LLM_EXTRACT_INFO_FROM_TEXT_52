import uvicorn
from aiogram import Dispatcher
from dotenv import load_dotenv
from pathlib import Path

from common.config import Settings
from core.db_service import DBService
from bots.rest_api.adapter import RestAdapter
from bots.rest_api.api import create_app
from core.rag.rag_service_wrapper import RagServiceWrapper

settings = Settings()
db = DBService()

def start_rest_api(rag_service):
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    settings = Settings()

    # Создание REST-адаптера и FastAPI приложения
    adapter = RestAdapter(rag_service)
    app = create_app(adapter)

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )

if __name__ == "__main__":
    start_rest_api(rag_service = RagServiceWrapper(settings, db))

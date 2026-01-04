from fastapi import FastAPI

from core.db import models
from core.db.session import engine
from core.rag.rag_service import RagService
from ml_service.api.routers import forward, history

__all__ = ["create_app"]


def create_app() -> FastAPI:
    app = FastAPI(title="ML Service", version="1.0.0")
    try:
        models.Base.metadata.create_all(bind=engine)
    except Exception:
        pass

    app.state.rag_service = RagService()
    app.include_router(forward.router)
    app.include_router(history.router)
    return app

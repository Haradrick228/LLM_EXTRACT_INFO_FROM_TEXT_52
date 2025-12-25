import base64
import io
import os
import time
import re
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status, Request
from PIL import Image
from pydantic import BaseModel

from core.db.session import session_scope
from core.db.repositories.request_log import RequestLogRepository
from core.rag.rag_service import RagService

JWT_SECRET = os.getenv("JWT_SECRET", "devsecret")
JWT_ALGO = os.getenv("JWT_ALGO", "HS256")
ADMIN_ROLE = "admin"


def create_app() -> FastAPI:
    app = FastAPI(title="ML Service", version="1.0.0")
    rag = RagService()

    def verify_admin(authorization: Optional[str] = Header(default=None)) -> None:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
        token = authorization.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
        if payload.get("role") != ADMIN_ROLE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    @app.post("/forward")
    async def forward(
        request: Request,
        image: Optional[UploadFile] = File(default=None),
        request_headers: Optional[str] = Header(default=None),
    ):
        start = time.perf_counter()
        input_type = "text"
        model_used = None
        status_tag = "ok"
        error = None
        text_len = None
        token_count = None
        image_w = None
        image_h = None
        resp_preview: Optional[str] = None

        def contains_cjk(s: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", s))

        def log_and_raise(code: int, message: str):
            nonlocal status_tag, error
            status_tag = "error"
            error = message
            _log_request(
                input_type=input_type,
                model=model_used,
                status=status_tag,
                error=error,
                start=start,
                text_len=text_len,
                token_count=token_count,
                image_width=image_w,
                image_height=image_h,
                response_preview=resp_preview,
            )
            raise HTTPException(status_code=code, detail=message)

        if image is None and request is None:
            log_and_raise(status.HTTP_400_BAD_REQUEST, "bad request")

        try:
            if image is not None:
                input_type = "image"
                img_bytes = await image.read()
                if not img_bytes:
                    log_and_raise(status.HTTP_400_BAD_REQUEST, "bad request")
                try:
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    image_w, image_h = pil_img.size
                except Exception:
                    log_and_raise(status.HTTP_403_FORBIDDEN, "модель не смогла обработать данные")
                # возвращаем то же изображение в base64
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                resp_preview = "image_base64"
                _log_request(
                    input_type=input_type,
                    model=model_used,
                    status=status_tag,
                    error=error,
                    start=start,
                    text_len=text_len,
                    token_count=token_count,
                    image_width=image_w,
                    image_height=image_h,
                    response_preview=resp_preview,
                )
                return {"image_base64": b64}

            # текстовый запрос (поддержка JSON и form-data без файла)
            data: Dict[str, Any] = {}
            try:
                content_type = request.headers.get("content-type", "")
                if content_type.startswith("application/json"):
                    data = await request.json()
                else:
                    form = await request.form()
                    data = dict(form)
            except Exception:
                log_and_raise(status.HTTP_400_BAD_REQUEST, "bad request")

            text = (data.get("text") or data.get("question") or "").strip()
            if not text:
                log_and_raise(status.HTTP_400_BAD_REQUEST, "bad request")
            text_len = len(text)
            token_count = len(text.split())
            model_used = data.get("model") or os.getenv("DEFAULT_MODEL", "deepseek")
            try:
                answer = rag.answer(text)
                # если ответ содержит CJK, пробуем еще пару раз
                retries = 2
                while retries > 0 and contains_cjk(answer):
                    answer = rag.answer(text)
                    retries -= 1
            except Exception:
                log_and_raise(status.HTTP_403_FORBIDDEN, "модель не смогла обработать данные")
            resp_preview = answer[:200]
            _log_request(
                input_type=input_type,
                model=model_used,
                status=status_tag,
                error=error,
                start=start,
                text_len=text_len,
                token_count=token_count,
                image_width=image_w,
                image_height=image_h,
                response_preview=resp_preview,
            )
            return {"answer": answer}
        except HTTPException:
            raise
        except Exception:
            log_and_raise(status.HTTP_403_FORBIDDEN, "модель не смогла обработать данные")

    @app.get("/history")
    async def history():
        with session_scope() as session:
            repo = RequestLogRepository(session)
            rows = repo.list(limit=200)
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "input_type": r.input_type,
                    "model": r.model,
                    "status": r.status,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                    "text_len": r.text_len,
                    "token_count": r.token_count,
                    "image_width": r.image_width,
                    "image_height": r.image_height,
                    "response_preview": r.response_preview,
                }
                for r in rows
            ]

    @app.delete("/history")
    async def clear_history(authorization: Optional[str] = Header(default=None)):
        verify_admin(authorization)
        with session_scope() as session:
            repo = RequestLogRepository(session)
            deleted = repo.clear_all()
            return {"deleted": deleted}

    @app.get("/stats")
    async def stats(authorization: Optional[str] = Header(default=None)):
        verify_admin(authorization)
        with session_scope() as session:
            repo = RequestLogRepository(session)
            return repo.stats()

    def _log_request(
        *,
        input_type: str,
        model: Optional[str],
        status: str,
        error: Optional[str],
        start: float,
        text_len: Optional[int],
        token_count: Optional[int],
        image_width: Optional[int],
        image_height: Optional[int],
        response_preview: Optional[str],
    ) -> None:
        duration_ms = (time.perf_counter() - start) * 1000
        with session_scope() as session:
            repo = RequestLogRepository(session)
            repo.add(
                input_type=input_type,
                model=model,
                status=status,
                error=error,
                duration_ms=duration_ms,
                text_len=text_len,
                token_count=token_count,
                image_width=image_width,
                image_height=image_height,
                response_preview=response_preview,
            )

    return app

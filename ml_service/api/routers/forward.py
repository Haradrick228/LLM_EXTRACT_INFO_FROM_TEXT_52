import base64
import io
import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile, status
from PIL import Image

from core.rag.rag_service import RagService
from ml_service.api.dependencies import get_rag_service
from ml_service.api.request_logging import log_request

router = APIRouter()

BAD_REQUEST_MSG = "Некорректный запрос"
PROCESSING_ERROR_MSG = "Не удалось обработать запрос."
IMAGE_ERROR_MSG = "Не удалось обработать изображение."


async def _read_request_payload(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        return await request.json()
    form = await request.form()
    return dict(form)


@router.post("/forward")
async def forward(
    request: Request,
    image: Optional[UploadFile] = File(default=None),
    request_headers: Optional[str] = Header(default=None),
    rag: RagService = Depends(get_rag_service),
):
    _ = request_headers  
    start = time.perf_counter()
    input_type = "text"
    model_used: Optional[str] = None
    status_tag = "ok"
    error: Optional[str] = None
    text_len: Optional[int] = None
    token_count: Optional[int] = None
    image_w: Optional[int] = None
    image_h: Optional[int] = None
    resp_preview: Optional[str] = None

    def log_and_raise(code: int, message: str):
        nonlocal status_tag, error
        status_tag = "error"
        error = message
        log_request(
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

    try:
        if image is not None:
            input_type = "image"
            img_bytes = await image.read()
            if not img_bytes:
                log_and_raise(status.HTTP_400_BAD_REQUEST, BAD_REQUEST_MSG)
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                image_w, image_h = pil_img.size
            except Exception:
                log_and_raise(status.HTTP_403_FORBIDDEN, IMAGE_ERROR_MSG)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            resp_preview = "image_base64"
            log_request(
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

        data: Dict[str, Any] = {}
        try:
            data = await _read_request_payload(request)
        except Exception:
            log_and_raise(status.HTTP_400_BAD_REQUEST, BAD_REQUEST_MSG)

        text = (data.get("text") or data.get("question") or "").strip()
        if not text:
            log_and_raise(status.HTTP_400_BAD_REQUEST, BAD_REQUEST_MSG)
        text_len = len(text)
        token_count = len(text.split())
        model_used = data.get("model") or os.getenv("DEFAULT_MODEL", "deepseek")
        try:
            answer = rag.answer_question(text, model_used)
        except Exception:
            log_and_raise(status.HTTP_403_FORBIDDEN, PROCESSING_ERROR_MSG)
        resp_preview = answer[:200]
        log_request(
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
        log_and_raise(status.HTTP_403_FORBIDDEN, PROCESSING_ERROR_MSG)

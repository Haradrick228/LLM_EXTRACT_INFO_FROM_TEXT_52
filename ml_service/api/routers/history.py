from fastapi import APIRouter, Depends

from ml_service.api.dependencies import verify_admin
from ml_service.api.request_logging import clear_history, get_history, get_stats

router = APIRouter()


@router.get("/history")
async def history():
    return get_history()


@router.delete("/history")
async def clear_history_endpoint(_: None = Depends(verify_admin)):
    deleted = clear_history()
    return {"deleted": deleted}


@router.get("/stats")
async def stats(_: None = Depends(verify_admin)):
    return get_stats()

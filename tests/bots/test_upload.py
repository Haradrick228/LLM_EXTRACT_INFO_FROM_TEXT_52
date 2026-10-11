"""Tests for UploadHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Dispatcher

from bots.telegram.handlers.upload import UploadHandler
from bots.telegram.texts import UPLOAD_FILES_BUTTON, UPLOAD_PROMPT_MESSAGE, UPLOAD_CANCELLED_MESSAGE, CANCEL_BUTTON


@pytest.fixture
def handler(mock_adapter, mock_upload_service, settings):
    return UploadHandler(mock_adapter, mock_upload_service, settings)


def _upd(text, uid=1, mid=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, uid, mid))


@pytest.mark.asyncio
async def test_admin_gets_upload_prompt(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(UPLOAD_FILES_BUTTON, uid=123))
    assert UPLOAD_PROMPT_MESSAGE in mock_adapter.send_message.call_args.kwargs.get("text", "")


@pytest.mark.asyncio
async def test_non_admin_blocked(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(UPLOAD_FILES_BUTTON, uid=999))
    mock_adapter.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_upload_cancel_returns_to_menu(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(UPLOAD_FILES_BUTTON, uid=123))
    await dp.feed_update(bot=mock_bot, update=_upd(CANCEL_BUTTON, uid=123, mid=2))
    texts = [c.kwargs.get("text", "") for c in mock_adapter.send_message.call_args_list]
    assert any(UPLOAD_CANCELLED_MESSAGE in t for t in texts)

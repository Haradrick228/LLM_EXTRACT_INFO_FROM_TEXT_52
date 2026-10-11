"""Tests for StartHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Dispatcher

from bots.telegram.handlers.start import StartHandler


@pytest.fixture
def handler(mock_adapter, settings):
    return StartHandler(mock_adapter, settings)


def _upd(text, uid=1, cid=1, u_id=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, cid, u_id))


@pytest.mark.asyncio
async def test_start_sends_welcome_and_menu(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd("/start"))
    mock_adapter.send_message.assert_called()
    assert "reply_markup" in mock_adapter.send_message.call_args.kwargs


@pytest.mark.asyncio
async def test_start_accepts_any_user(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd("/start", uid=999, cid=999))
    mock_adapter.send_message.assert_called()

"""Tests for BackHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Dispatcher

from bots.telegram.handlers.back import BackHandler
from bots.telegram.texts import BACK_BUTTON, BACK_RETURN_MESSAGE


@pytest.fixture
def handler(mock_adapter, settings):
    return BackHandler(mock_adapter, settings)


def _upd(text, uid=1, cid=1, u_id=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, cid, u_id))


@pytest.mark.asyncio
async def test_back_returns_to_menu(handler, mock_adapter, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(BACK_BUTTON))
    mock_adapter.send_message.assert_called()
    assert BACK_RETURN_MESSAGE in str(mock_adapter.send_message.call_args.kwargs.get("text", ""))
    assert "reply_markup" in mock_adapter.send_message.call_args.kwargs

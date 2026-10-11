"""Tests for NPSHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage, StorageKey
from aiogram import Dispatcher

from bots.telegram.handlers.nps import NPSHandler
from bots.telegram.handlers.states import MenuState


@pytest.fixture
def handler(mock_adapter, mock_qa):
    return NPSHandler(mock_adapter, mock_qa)


def _upd(text, uid=1, mid=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, uid, mid))


def _cb(data, uid=1, mid=1):
    from conftest import make_update_from_dict, make_callback_dict
    return make_update_from_dict(make_callback_dict(data, uid, uid, mid))


@pytest.mark.asyncio
async def test_nps_saves_with_comment(handler, mock_qa, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    # Set state to waiting_for_nps_rating
    key = StorageKey(bot_id=mock_bot.id, chat_id=1, user_id=1)
    await storage.set_state(key, MenuState.waiting_for_nps_rating)
    await dp.feed_update(bot=mock_bot, update=_cb("nps:10", mid=1))
    await dp.feed_update(bot=mock_bot, update=_upd("Отличный бот!", mid=2))
    mock_qa.save_nps.assert_called()
    assert mock_qa.save_nps.call_args.kwargs.get("score") == 10


@pytest.mark.asyncio
async def test_nps_skip_comment(handler, mock_qa, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    key = StorageKey(bot_id=mock_bot.id, chat_id=1, user_id=1)
    await storage.set_state(key, MenuState.waiting_for_nps_rating)
    await dp.feed_update(bot=mock_bot, update=_cb("nps:7", mid=1))
    await dp.feed_update(bot=mock_bot, update=_cb("nps_skip_comment", mid=2))
    mock_qa.save_nps.assert_called()
    assert mock_qa.save_nps.call_args.kwargs.get("score") == 7

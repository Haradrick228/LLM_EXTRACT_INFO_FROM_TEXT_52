"""Tests for LibraryHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Dispatcher

from bots.telegram.handlers.library import LibraryHandler
from bots.telegram.texts import FILE_LIBRARY_BUTTON, FILE_LIBRARY_EMPTY, FILE_LIBRARY_PROMPT, FILE_LIBRARY_UNKNOWN_NUMBER


@pytest.fixture
def handler(mock_adapter, mock_file_library, mock_rag):
    return LibraryHandler(mock_adapter, mock_file_library, mock_rag)


def _upd(text, uid=1, mid=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, uid, mid))


def _all_texts(adapter):
    """Collect all texts sent through adapter.send_message (positional or kwarg)."""
    result = []
    for call in adapter.send_message.call_args_list:
        args, kwargs = call
        text = kwargs.get("text", "") or (args[1] if len(args) > 1 else "")
        result.append(text)
    return result


@pytest.mark.asyncio
async def test_library_shows_file_list(handler, mock_adapter, mock_file_library, mock_bot):
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(FILE_LIBRARY_BUTTON))
    texts = _all_texts(mock_adapter)
    assert any(FILE_LIBRARY_PROMPT in t for t in texts)


@pytest.mark.asyncio
async def test_library_empty_state(handler, mock_adapter, mock_file_library, mock_bot):
    mock_file_library.get_page.return_value = ([], None, None)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(FILE_LIBRARY_BUTTON))
    texts = _all_texts(mock_adapter)
    assert any(FILE_LIBRARY_EMPTY in t for t in texts)


@pytest.mark.asyncio
async def test_library_invalid_number(handler, mock_adapter, mock_rag, mock_bot):
    mock_rag.list_files.return_value = ["file1.pdf"]
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(FILE_LIBRARY_BUTTON))
    await dp.feed_update(bot=mock_bot, update=_upd("999", mid=2))
    texts = _all_texts(mock_adapter)
    assert any(FILE_LIBRARY_UNKNOWN_NUMBER in t for t in texts)


@pytest.mark.asyncio
async def test_library_valid_number_sends_file(handler, mock_adapter, mock_rag, mock_bot):
    mock_rag.list_files.return_value = ["file1.pdf", "file2.pdf"]
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(FILE_LIBRARY_BUTTON))
    await dp.feed_update(bot=mock_bot, update=_upd("2", mid=2))
    mock_adapter.send_file.assert_called()
    assert mock_adapter.send_file.call_args[0][1] == "file2.pdf"

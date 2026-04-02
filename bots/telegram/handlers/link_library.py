from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bots.telegram.adapters.adapter import TelegramAdapter
from bots.telegram.texts import LINK_LIBRARY_BUTTON, LINK_LIBRARY_NEXT, LINK_LIBRARY_PREV
from core.services.link_library_service import LinkLibraryService


class LinkLibraryHandler:
    def __init__(self, adapter: TelegramAdapter, link_library: LinkLibraryService):
        self.adapter = adapter
        self.link_library = link_library
        self.router = Router(name="link-library")

        self.router.message.register(self.show_page, F.text == LINK_LIBRARY_BUTTON)
        self.router.message.register(self.show_page, Command(commands=["links"]))
        self.router.callback_query.register(self.on_page_callback, F.data.startswith("page:"))

    async def on_page_callback(self, query: CallbackQuery) -> None:
        page = int(query.data.split(":", 1)[1])
        await self._render_page(chat_id=query.message.chat.id, page=page, callback=query)

    async def show_page(self, message: Message) -> None:
        await self._render_page(chat_id=message.chat.id, page=1)

    async def _render_page(
        self,
        chat_id: int,
        page: int,
        callback: CallbackQuery | None = None,
    ) -> None:
        page_links, prev_page, next_page = self.link_library.get_page(page)

        if not page_links:
            text = "No links have been added yet."
            if callback:
                await callback.message.edit_text(text)
                await callback.answer()
            else:
                await self.adapter.bot.send_message(chat_id, text)
            return

        message_body = "\n".join(
            f"{resource_id}. <a href=\"{url}\">{url}</a>"
            for resource_id, url in page_links
        )

        keyboard = InlineKeyboardBuilder()
        nav_buttons = []
        if prev_page is not None:
            nav_buttons.append(
                InlineKeyboardButton(text=LINK_LIBRARY_PREV, callback_data=f"page:{prev_page}")
            )
        if next_page is not None:
            nav_buttons.append(
                InlineKeyboardButton(text=LINK_LIBRARY_NEXT, callback_data=f"page:{next_page}")
            )
        if nav_buttons:
            keyboard.row(*nav_buttons)

        if callback:
            await callback.message.edit_text(
                message_body,
                reply_markup=keyboard.as_markup(),
                parse_mode="HTML",
            )
            await callback.answer()
            return

        await self.adapter.bot.send_message(
            chat_id,
            message_body,
            reply_markup=keyboard.as_markup(),
            parse_mode="HTML",
        )

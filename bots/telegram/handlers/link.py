from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import ADD_RESOURCE_BUTTON, BACK_BUTTON, LINK_CONFIRMATION_TEMPLATE, LINK_RESOURCE_PROMPT
from bots.telegram.utils import is_admin
from common.config import Settings
from common.interfaces.user_interaction import UserInteraction


class LinkHandler:
    """Добавление ссылок в базу знаний (только для администраторов)."""
    def __init__(self, adapter: UserInteraction, link_service, settings: Settings) -> None:
        self.adapter = adapter
        self.link_service = link_service
        self.settings = settings
        self.router = Router(name="link")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.message(F.text == ADD_RESOURCE_BUTTON)
        async def prompt_link(message, state: FSMContext):
            if not is_admin(self.settings, message.from_user.id, message.from_user.username):
                return

            await state.set_state(MenuState.add_link)
            keyboard = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
                resize_keyboard=True,
            )
            await self.adapter.send_message(
                user_id=message.from_user.id,
                text=LINK_RESOURCE_PROMPT,
                reply_markup=keyboard,
            )

        @self.router.message(StateFilter(MenuState.add_link), F.text.startswith("http"))
        async def handle_link(message, state: FSMContext):
            if not is_admin(self.settings, message.from_user.id, message.from_user.username):
                await state.clear()
                return

            url = message.text.strip()
            await self.adapter.send_message(
                user_id=message.from_user.id,
                text=LINK_CONFIRMATION_TEMPLATE.format(url=url),
            )
            success, info = await self.link_service.add_link(url)
            await self.adapter.send_message(user_id=message.from_user.id, text=info)
            await state.clear()

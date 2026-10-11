from aiogram import Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

from bots.telegram.handlers.menu import get_main_menu
from bots.telegram.texts import WELCOME_MESSAGE
from common.config import Settings
from common.interfaces.user_interaction import UserInteraction


class StartHandler:
    """Отправляет приветственное сообщение и главное меню по команде /start.

    Очищает состояние FSM и выводит клавиатуру, соответствующую роли
    пользователя (администратор или обычный пользователь).
    """
    def __init__(self, adapter: UserInteraction, settings: Settings) -> None:
        self.adapter = adapter
        self.settings = settings
        self.router = Router(name="start")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.message(Command("start"))
        async def _(message, state: FSMContext):  # noqa: WPS430 - inline handler
            await state.clear()
            await self.adapter.send_message(
                user_id=message.from_user.id,
                text=WELCOME_MESSAGE,
                reply_markup=get_main_menu(
                    self.settings,
                    message.from_user.id,
                    message.from_user.username,
                ),
            )

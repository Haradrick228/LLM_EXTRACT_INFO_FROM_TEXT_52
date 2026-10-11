from aiogram import F, Router
from aiogram.fsm.context import FSMContext

from bots.telegram.handlers.menu import get_main_menu
from bots.telegram.texts import BACK_BUTTON, BACK_RETURN_MESSAGE
from common.config import Settings
from common.interfaces.user_interaction import UserInteraction


class BackHandler:
    """Handler that returns the user to the main menu.

    Matches the BACK_BUTTON text, clears FSM state and sends the main
    menu keyboard.
    """
    def __init__(self, adapter: UserInteraction, settings: Settings) -> None:
        self.adapter = adapter
        self.settings = settings
        self.router = Router(name="back")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.message(F.text == BACK_BUTTON)
        async def _(message, state: FSMContext):  # noqa: WPS430 - inline handler registration
            await state.clear()
            await self.adapter.send_message(
                user_id=message.from_user.id,
                text=BACK_RETURN_MESSAGE,
                reply_markup=get_main_menu(
                    self.settings,
                    message.from_user.id,
                    message.from_user.username,
                ),
            )

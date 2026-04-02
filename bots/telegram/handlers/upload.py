from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from bots.telegram.handlers.menu import get_main_menu
from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import (
    CANCEL_BUTTON,
    UPLOAD_FILES_BUTTON,
    UPLOAD_PROMPT_MESSAGE,
)
from bots.telegram.utils import is_admin
from common.interfaces.user_interaction import UserInteraction

if TYPE_CHECKING:
    from common.config import Settings
    from core.services.upload_service import UploadService


UPLOAD_DISABLED_TEXT = (
    "Uploading is disabled. Set FOLDER_PATH in the environment to enable it."
)
CANCELLED_MESSAGE = "Upload cancelled. Returning to the main menu."


class UploadHandler:
    def __init__(
        self,
        adapter: UserInteraction,
        upload_service: UploadService,
        settings: Settings,
    ) -> None:
        self.adapter = adapter
        self.upload_service = upload_service
        self.settings = settings
        self.router = Router(name="upload")
        self._register_routes()

    def _register_routes(self) -> None:
        cancel_keyboard = ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text=CANCEL_BUTTON)]],
            resize_keyboard=True,
        )

        @self.router.message(F.text == UPLOAD_FILES_BUTTON)
        async def prompt_upload(message, state: FSMContext):
            if not is_admin(self.settings, message.from_user.id, message.from_user.username):
                return

            await state.set_state(MenuState.upload)
            await self.adapter.send_message(
                user_id=message.from_user.id,
                text=UPLOAD_PROMPT_MESSAGE,
                reply_markup=cancel_keyboard,
            )

        @self.router.message(
            StateFilter(MenuState.upload),
            F.document.file_name.regexp(r".+\.(pdf|docx|pptx)$", flags=re.IGNORECASE),
        )
        async def handle_upload(message, state: FSMContext):
            user_id = message.from_user.id
            username = message.from_user.username

            if not is_admin(self.settings, user_id, username):
                await state.clear()
                return

            destination_root = self._ensure_upload_root()
            if not destination_root:
                await self.adapter.send_message(user_id=user_id, text=UPLOAD_DISABLED_TEXT)
                await state.clear()
                return

            destination_path = os.path.join(destination_root, message.document.file_name)
            downloaded_path = await self.adapter.download_file(
                message.document.file_id,
                destination_path,
            )

            _, info = self.upload_service.add_file(downloaded_path)
            await self.adapter.send_message(user_id=user_id, text=info)
            await state.clear()

        @self.router.message(StateFilter(MenuState.upload), F.text == CANCEL_BUTTON)
        async def cancel_upload(message, state: FSMContext):
            user_id = message.from_user.id
            username = message.from_user.username

            if not is_admin(self.settings, user_id, username):
                await state.clear()
                return

            await state.clear()
            await self.adapter.send_message(
                user_id=user_id,
                text=CANCELLED_MESSAGE,
                reply_markup=get_main_menu(self.settings, user_id, username),
            )

    def _ensure_upload_root(self) -> str:
        root = self.upload_service.upload_folder
        if not root:
            return ""
        os.makedirs(root, exist_ok=True)
        return root

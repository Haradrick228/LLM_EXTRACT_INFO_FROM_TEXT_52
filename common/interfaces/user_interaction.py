"""Abstract interface for user interaction (bot adapters)."""
from abc import ABC, abstractmethod
from typing import Any, Optional


class UserInteraction(ABC):
    """Abstract interface for bot adapters to interact with users."""

    @abstractmethod
    async def send_message(self, user_id: str, text: str, **kwargs: Any) -> None:
        """Send a text message to a user."""
        pass

    @abstractmethod
    async def send_file(self, user_id: str, file_path: str, **kwargs: Any) -> None:
        """Send a file to a user."""
        pass

    @abstractmethod
    async def download_file(self, file_id: str, destination: str) -> str:
        """Download a file from the user to the specified destination.

        Returns:
            The path to the downloaded file.
        """
        pass


class BotAdapter(ABC):
    """Abstract interface for bot adapters that handle messages."""

    @abstractmethod
    async def handle_message(self, text: str) -> str:
        """Handle an incoming message and return a response."""
        pass

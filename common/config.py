"""Settings module for the application."""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Bot settings
    bot_token: str = field(default_factory=lambda: os.getenv("BOT_TOKEN", ""))

    # Admin settings
    admin_ids: List[int] = field(default_factory=lambda: [
        int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()
    ])
    admin_usernames: List[str] = field(default_factory=lambda: [
        x.strip().lstrip("@") for x in os.getenv("ADMIN_USERNAMES", "").split(",") if x.strip()
    ])

    # Folder path for file scanning
    folder_path: str = field(default_factory=lambda: os.getenv("FOLDER_PATH", ""))

    # API settings
    API_HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8001")))

    # Database settings (optional, used by some components)
    POSTGRES_HOST: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    POSTGRES_PORT: str = field(default_factory=lambda: os.getenv("POSTGRES_PORT", "5433"))
    POSTGRES_DB: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", ""))
    POSTGRES_USER: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", ""))
    POSTGRES_PASSWORD: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    # ChromaDB settings
    CHROMA_SERVER_HOST: str = field(default_factory=lambda: os.getenv("CHROMA_SERVER_HOST", "localhost"))
    CHROMA_SERVER_HTTP_PORT: int = field(default_factory=lambda: int(os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")))

    # RAG settings
    client_id: str = field(default_factory=lambda: os.getenv("CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("CLIENT_SECRET", ""))
    scope: str = field(default_factory=lambda: os.getenv("SCOPE", "GIGACHAT_API_PERS"))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gigachat"))
    persist_directory: str = field(default_factory=lambda: os.getenv("PERSIST_DIRECTORY", "./chroma_persist"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

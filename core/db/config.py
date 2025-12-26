import os
from dataclasses import dataclass

DEFAULT_SQLITE_URL = "sqlite:///./ml_service.db"


@dataclass
class DBSettings:
    url: str
    echo: bool = False

    @staticmethod
    def load() -> "DBSettings":
        url = os.getenv("DATABASE_URL")
        if not url:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5433")
            db = os.getenv("POSTGRES_DB", "")
            user = os.getenv("POSTGRES_USER", "")
            password = os.getenv("POSTGRES_PASSWORD", "")
            if host and db and user:
                url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
            else:
                url = DEFAULT_SQLITE_URL
        echo = os.getenv("SQL_ECHO", "0").lower() in ("1", "true", "yes")
        return DBSettings(url=url, echo=echo)

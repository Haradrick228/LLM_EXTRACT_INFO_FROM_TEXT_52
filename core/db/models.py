from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    input_type = Column(String(32), nullable=False)
    model = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False)
    error = Column(String(512), nullable=True)
    duration_ms = Column(Float, nullable=True)
    text_len = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    response_preview = Column(String(1024), nullable=True)

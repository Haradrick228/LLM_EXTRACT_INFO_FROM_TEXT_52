"""create request_log table

Revision ID: 0001_request_log
Revises:
Create Date: 2025-12-26
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_request_log"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "request_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("input_type", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128)),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.String(length=512)),
        sa.Column("duration_ms", sa.Float),
        sa.Column("text_len", sa.Integer),
        sa.Column("token_count", sa.Integer),
        sa.Column("image_width", sa.Integer),
        sa.Column("image_height", sa.Integer),
        sa.Column("response_preview", sa.String(length=1024)),
    )


def downgrade() -> None:
    op.drop_table("request_log")

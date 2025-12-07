from datetime import datetime

from sqlalchemy import BigInteger, Text, Boolean, DateTime, text
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.infrastructure.db.base import Base


class QaPairORM(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    is_generated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

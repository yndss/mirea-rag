from datetime import datetime
import uuid

from sqlalchemy import BigInteger, Text, Boolean, DateTime, Numeric, Float, String, JSON, ForeignKey, text, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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


class RagRunORM(Base):
    __tablename__ = "rag_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    retriever_top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(16), nullable=False)
    context_text: Mapped[str] = mapped_column(Text, nullable=False)
    final_prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    extra_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    usage_prompt_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    usage_completion_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    usage_total_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(20, 10), nullable=True)
    latency_ms_total: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    latency_ms_retrieval: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    latency_ms_llm: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    latency_ms_embedding: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class RagRunHitORM(Base):
    __tablename__ = "rag_run_hits"

    rag_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    qa_pair_id: Mapped[int | None] = mapped_column(
        ForeignKey("qa_pairs.id", ondelete="SET NULL"),
        nullable=True,
    )
    distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_in_context: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

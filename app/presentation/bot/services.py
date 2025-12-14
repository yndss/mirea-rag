from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger

from app.application.rag_service import RagService
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient
from sqlalchemy.ext.asyncio import AsyncSession


def _build_rag_service(session: AsyncSession) -> RagService:
    qa_repo = SqlAlchemyQaPairRepository(session)
    embedding_provider = OpenRouterEmbeddingProvider()
    llm_client = OpenRouterLlmClient()
    return RagService(
        qa_repo=qa_repo,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
    )


@asynccontextmanager
async def rag_service_context() -> AsyncIterator[RagService]:
    async with SessionLocal() as session:
        logger.debug("Opened database session for Telegram request")
        try:
            rag_service = _build_rag_service(session)
            yield rag_service
            await session.commit()
            logger.debug("Session commit completed")
        except Exception as exc:
            logger.exception(
                "Error during RAG pipeline execution, rolling back session: {}",
                exc,
            )
            await session.rollback()
            raise
        finally:
            logger.debug("Database session closed")

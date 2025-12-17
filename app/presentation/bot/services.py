import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from loguru import logger

from app.application.rag_service import RagService
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.db.rag_run_repository import SqlAlchemyRagRunRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient
from sqlalchemy.ext.asyncio import AsyncSession


_shared_clients_lock = asyncio.Lock()
_shared_embedding_provider: OpenRouterEmbeddingProvider | None = None
_shared_llm_client: OpenRouterLlmClient | None = None


async def _get_shared_clients() -> tuple[OpenRouterEmbeddingProvider, OpenRouterLlmClient]:
    global _shared_embedding_provider, _shared_llm_client

    if _shared_embedding_provider is not None and _shared_llm_client is not None:
        return _shared_embedding_provider, _shared_llm_client

    async with _shared_clients_lock:
        if _shared_embedding_provider is None:
            _shared_embedding_provider = OpenRouterEmbeddingProvider()
        if _shared_llm_client is None:
            _shared_llm_client = OpenRouterLlmClient()
        return _shared_embedding_provider, _shared_llm_client


async def init_shared_clients(**_: object) -> None:
    await _get_shared_clients()


async def close_shared_clients(**_: object) -> None:
    global _shared_embedding_provider, _shared_llm_client

    async with _shared_clients_lock:
        embedding_provider, _shared_embedding_provider = _shared_embedding_provider, None
        llm_client, _shared_llm_client = _shared_llm_client, None

    if embedding_provider is not None:
        try:
            await embedding_provider.close()
        except Exception as exc:
            logger.exception("Failed to close embedding client: {}", exc)

    if llm_client is not None:
        try:
            await llm_client.close()
        except Exception as exc:
            logger.exception("Failed to close LLM client: {}", exc)


def _build_rag_service(
    session: AsyncSession,
    embedding_provider: OpenRouterEmbeddingProvider,
    llm_client: OpenRouterLlmClient,
) -> RagService:
    qa_repo = SqlAlchemyQaPairRepository(session)
    run_repo = SqlAlchemyRagRunRepository(session)
    return RagService(
        qa_repo=qa_repo,
        embedding_provider=embedding_provider,
        llm_client=llm_client,
        run_repo=run_repo,
    )


@asynccontextmanager
async def rag_service_context() -> AsyncIterator[RagService]:
    embedding_provider, llm_client = await _get_shared_clients()

    async with SessionLocal() as session:
        logger.debug("Opened database session for Telegram request")
        try:
            rag_service = _build_rag_service(
                session,
                embedding_provider=embedding_provider,
                llm_client=llm_client,
            )
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

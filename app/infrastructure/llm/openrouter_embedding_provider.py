from typing import Sequence
import math

from loguru import logger
from openai import AsyncOpenAI

from app.infrastructure.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_TIMEOUT,
    OPENROUTER_API_KEY,
)


class OpenRouterEmbeddingProvider:

    def __init__(
        self,
        *,
        model_name: str | None = None,
        base_url: str = EMBEDDING_BASE_URL,
        api_key: str | None = OPENROUTER_API_KEY,
        timeout: float = EMBEDDING_TIMEOUT,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        resolved_model_name = model_name or EMBEDDING_MODEL_NAME
        if not resolved_model_name:
            raise RuntimeError("EMBEDDING_MODEL_NAME is not set")

        self._model = resolved_model_name
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def embed(self, text: str) -> Sequence[float]:
        embeddings = await self.embed_many([text])
        return embeddings[0]

    async def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []

        logger.debug(
            "Requesting embeddings via OpenRouter (model={}, count={})",
            self._model,
            len(texts),
        )
        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=list(texts),
                encoding_format="float",
            )
            embeddings: list[list[float]] = [
                self._l2_normalize(item.embedding) for item in response.data
            ]
            logger.debug("Embeddings received (items={})", len(embeddings))
            return embeddings
        except Exception as exc:
            logger.exception("Embedding request to OpenRouter failed: {}", exc)
            raise

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _l2_normalize(vec: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return list(vec)
        return [x / norm for x in vec]

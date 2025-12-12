from typing import Sequence

from loguru import logger
from openai import AsyncOpenAI

from app.infrastructure.config import OPENROUTER_API_KEY, EMBEDDING_MODEL_NAME


class OpenRouterEmbeddingProvider:

    def __init__(
        self,
        timeout: float = 30.0,
        base_url: str = "https://openrouter.ai/api/v1",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not EMBEDDING_MODEL_NAME:
            raise RuntimeError("EMBEDDING_MODEL_NAME is not set")

        self._model = EMBEDDING_MODEL_NAME
        self._extra_headers = extra_headers
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=OPENROUTER_API_KEY,
            default_headers=extra_headers,
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
            embeddings: list[list[float]] = [item.embedding for item in response.data]
            logger.debug("Embeddings received (items={})", len(embeddings))
            return embeddings
        except Exception as exc:
            logger.exception("Embedding request to OpenRouter failed: {}", exc)
            raise

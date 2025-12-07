from typing import Sequence
import requests

from app.infrastructure.config import OPENROUTER_API_KEY, EMBEDDING_MODEL_NAME


class OpenRouterEmbeddingProvider:

    def __init__(
        self,
        timeout: float = 30.0,
        base_url: str = "https://openrouter.ai/api/v1/embeddings",
    ) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not EMBEDDING_MODEL_NAME:
            raise RuntimeError("EMBEDDING_MODEL_NAME is not set")

        self._timeout = timeout
        self._base_url = base_url
        self._api_key = OPENROUTER_API_KEY
        self._model = EMBEDDING_MODEL_NAME

    def embed(self, text: str) -> Sequence[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []

        response = requests.post(
            self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "input": list(texts),
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings: list[list[float]] = [item["embedding"] for item in data["data"]]
        return embeddings

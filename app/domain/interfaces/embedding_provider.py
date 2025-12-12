from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> Sequence[float]: ...

    async def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

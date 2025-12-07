from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...
    
    def embed_many(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...
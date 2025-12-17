from typing import Protocol, Sequence
from app.domain.models.qa_pair import QaPair, QaPairHit


class QaPairRepository(Protocol):
    async def add(self, qa: QaPair) -> QaPair: ...

    async def add_many(self, qa_list: Sequence[QaPair]) -> None: ...

    async def list_all(self) -> Sequence[QaPair]: ...

    async def find_top_k(
        self,
        query_embedding: Sequence[float],
        k: int,
    ) -> Sequence[QaPairHit]: ...

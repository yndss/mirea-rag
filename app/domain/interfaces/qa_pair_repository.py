from typing import Protocol, Sequence
from app.domain.models.qa_pair import QaPair


class QaPairRepository(Protocol):
    def add(self, qa: QaPair) -> QaPair: ...

    def add_many(self, qa_list: Sequence[QaPair]) -> None: ...

    def list_all(self) -> Sequence[QaPair]: ...

    def find_top_k(
        self,
        query_embedding: Sequence[float],
        k: int,
    ) -> Sequence[QaPair]: ...

from typing import Protocol, Sequence
from uuid import UUID

from app.domain.models.rag_run import RagRun, RagRunHit


class RagRunRepository(Protocol):
    async def add_run(self, run: RagRun, hits: Sequence[RagRunHit]) -> UUID: ...

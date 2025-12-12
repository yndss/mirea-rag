from typing import Sequence

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.interfaces.qa_pair_repository import QaPairRepository
from app.domain.models.qa_pair import QaPair
from app.infrastructure.db.models import QaPairORM


class SqlAlchemyQaPairRepository(QaPairRepository):

    def __init__(self, session: AsyncSession) -> None:
        self._session: AsyncSession = session

    @staticmethod
    def _to_domain(row: QaPairORM) -> QaPair:
        return QaPair(
            id=row.id,
            question=row.question,
            answer=row.answer,
            source_url=row.source_url,
            topic=row.topic,
            is_generated=row.is_generated,
            embedding=list(row.embedding),
            created_at=row.created_at,
        )

    async def add(self, qa: QaPair) -> QaPair:
        orm_obj = QaPairORM(
            question=qa.question,
            answer=qa.answer,
            source_url=qa.source_url,
            topic=qa.topic,
            is_generated=qa.is_generated,
            embedding=list(qa.embedding),
        )
        self._session.add(orm_obj)
        await self._session.flush()
        await self._session.refresh(orm_obj)

        qa.id = orm_obj.id
        qa.created_at = orm_obj.created_at
        logger.debug("Inserted QA pair (id={}, topic={})", qa.id, qa.topic)
        return qa

    async def add_many(self, qa_list: Sequence[QaPair]) -> None:
        if not qa_list:
            return

        for qa in qa_list:
            orm_obj = QaPairORM(
                question=qa.question,
                answer=qa.answer,
                source_url=qa.source_url,
                topic=qa.topic,
                is_generated=qa.is_generated,
                embedding=list(qa.embedding),
            )
            self._session.add(orm_obj)

        await self._session.flush()
        logger.info("Inserted batch of QA pairs (count={})", len(qa_list))

    async def list_all(self) -> Sequence[QaPair]:
        result = await self._session.scalars(select(QaPairORM))
        rows = result.all()
        logger.debug("Fetched all QA pairs (count={})", len(rows))
        return [self._to_domain(row) for row in rows]

    async def find_top_k(
        self, query_embedding: Sequence[float], k: int
    ) -> Sequence[QaPair]:
        stmt = (
            select(QaPairORM)
            .order_by(QaPairORM.embedding.l2_distance(list(query_embedding)))
            .limit(k)
        )
        result = await self._session.scalars(stmt)
        rows = result.all()
        logger.debug("Vector search returned {} items (k={})", len(rows), k)
        return [self._to_domain(row) for row in rows]

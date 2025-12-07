from typing import Sequence
from sqlalchemy.orm import Session

from app.domain.models.qa_pair import QaPair
from app.domain.interfaces.qa_pair_repository import QaPairRepository
from app.infrastructure.db.models import QaPairORM 


class SqlAlchemyQaPairRepository(QaPairRepository):
    

    def __init__(self, session: Session) -> None:
        self._session = session

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
            created_at=row.created_at
        )
    
    def add(self, qa: QaPair) -> QaPair:
        orm_obj = QaPairORM(
            question=qa.question,
            answer=qa.answer,
            source_url=qa.source_url,
            topic=qa.topic,
            is_generated=qa.is_generated,
            embedding=list(qa.embedding)
        )
        self._session.add(orm_obj)
        self._session.flush()

        qa.id = orm_obj.id
        qa.created_at = orm_obj.created_at
        return qa

    def add_many(self, qa_list: Sequence[QaPair]) -> None:
        if not qa_list:
            return
        
        for qa in qa_list:
            orm_obj = QaPairORM(
                question=qa.question,
                answer=qa.answer,
                source_url=qa.source_url,
                topic=qa.topic,
                is_generated=qa.is_generated,
                embedding=list(qa.embedding)
            )
            self._session.add(orm_obj)
        
        self._session.flush()
    
    def list_all(self) -> Sequence[QaPair]:
        rows: list[QaPairORM] = self._session.query(QaPairORM).all()
        return [self._to_domain(row) for row in rows]
    
    def find_top_k(
        self,
        query_embedding: Sequence[float],
        k: int
    ) -> Sequence[QaPair]:
        rows: list[QaPairORM] = (
            self._session
            .query(QaPairORM)
            .order_by(
                QaPairORM.embedding.l2_distance(list(query_embedding))
            )
            .limit(k)
            .all()
        )
        return [self._to_domain(row) for row in rows]
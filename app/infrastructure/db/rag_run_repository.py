import uuid
from typing import Sequence
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.interfaces.rag_run_repository import RagRunRepository
from app.domain.models.rag_run import RagRun, RagRunHit
from app.infrastructure.db.models import RagRunHitORM, RagRunORM


class SqlAlchemyRagRunRepository(RagRunRepository):

    def __init__(self, session: AsyncSession) -> None:
        self._session: AsyncSession = session

    async def add_run(self, run: RagRun, hits: Sequence[RagRunHit]) -> UUID:
        run_id = run.id or uuid.uuid4()

        orm_run = RagRunORM(
            id=run_id,
            user_id=run.user_id,
            question_text=run.question_text,
            retriever_top_k=run.retriever_top_k,
            similarity_threshold=run.similarity_threshold,
            distance_metric=run.distance_metric,
            context_text=run.context_text,
            final_prompt_text=run.final_prompt_text,
            model_name=run.model_name,
            temperature=run.temperature,
            extra_params=run.extra_params,
            answer_text=run.answer_text,
            usage_prompt_tokens=run.usage_prompt_tokens,
            usage_completion_tokens=run.usage_completion_tokens,
            usage_total_tokens=run.usage_total_tokens,
            cost_usd=run.cost_usd,
            latency_ms_total=run.latency_ms_total,
            latency_ms_retrieval=run.latency_ms_retrieval,
            latency_ms_llm=run.latency_ms_llm,
            latency_ms_embedding=run.latency_ms_embedding,
        )
        self._session.add(orm_run)
        await self._session.flush()

        for hit in hits:
            orm_hit = RagRunHitORM(
                rag_run_id=run_id,
                rank=hit.rank,
                qa_pair_id=hit.qa_pair_id,
                distance=hit.distance,
                similarity=hit.similarity,
                used_in_context=hit.used_in_context,
            )
            self._session.add(orm_hit)

        await self._session.flush()
        return run_id

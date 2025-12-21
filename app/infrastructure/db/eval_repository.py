from __future__ import annotations

from typing import Sequence
from uuid import UUID

from loguru import logger
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.eval import EvalCase, EvalDataset, EvalResult, EvalRun
from app.infrastructure.db.models import (
    EvalCaseORM,
    EvalDatasetORM,
    EvalResultORM,
    EvalRunORM,
)


class SqlAlchemyEvalRepository:

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _to_dataset(row: EvalDatasetORM) -> EvalDataset:
        return EvalDataset(
            id=row.id,
            name=row.name,
            description=row.description,
            created_at=row.created_at,
        )

    @staticmethod
    def _to_case(row: EvalCaseORM) -> EvalCase:
        return EvalCase(
            dataset_id=row.dataset_id,
            case_id=row.case_id,
            question_text=row.question_text,
            ideal_answer_text=row.ideal_answer_text,
            meta_json=row.meta_json,
        )

    @staticmethod
    def _to_run(row: EvalRunORM) -> EvalRun:
        return EvalRun(
            id=row.id,
            dataset_id=row.dataset_id,
            created_at=row.created_at,
            system_version=row.system_version,
            retriever_config_json=row.retriever_config_json,
            llm_config_json=row.llm_config_json,
        )

    @staticmethod
    def _to_result(row: EvalResultORM) -> EvalResult:
        return EvalResult(
            eval_run_id=row.eval_run_id,
            case_id=row.case_id,
            model_answer_text=row.model_answer_text,
            bert_score=row.bert_score,
            precision=row.precision,
            recall=row.recall,
            f1=row.f1,
            rouge_1=row.rouge_1,
            rouge_l=row.rouge_l,
            llm_judge_score=row.llm_judge_score,
            latency_ms=row.latency_ms,
            cost_usd=float(row.cost_usd) if row.cost_usd is not None else None,
            tokens_total=row.tokens_total,
        )

    async def get_dataset_by_name(self, name: str) -> EvalDataset | None:
        row = await self._session.scalar(
            select(EvalDatasetORM).where(EvalDatasetORM.name == name)
        )
        return self._to_dataset(row) if row is not None else None

    async def create_dataset(self, name: str, description: str | None) -> EvalDataset:
        obj = EvalDatasetORM(name=name, description=description)
        self._session.add(obj)
        await self._session.flush()
        await self._session.refresh(obj)
        logger.info("Created eval dataset (id={}, name={})", obj.id, obj.name)
        return self._to_dataset(obj)

    async def get_or_create_dataset(
        self, name: str, description: str | None
    ) -> EvalDataset:
        dataset = await self.get_dataset_by_name(name)
        if dataset is not None:
            return dataset
        return await self.create_dataset(name, description)

    async def replace_cases(self, dataset_id: int, cases: Sequence[EvalCase]) -> None:
        await self._session.execute(
            delete(EvalCaseORM).where(EvalCaseORM.dataset_id == dataset_id)
        )

        for case in cases:
            self._session.add(
                EvalCaseORM(
                    dataset_id=dataset_id,
                    case_id=case.case_id,
                    question_text=case.question_text,
                    ideal_answer_text=case.ideal_answer_text,
                    meta_json=case.meta_json,
                )
            )

        await self._session.flush()
        logger.info(
            "Replaced eval cases (dataset_id={}, count={})",
            dataset_id,
            len(cases),
        )

    async def list_cases(self, dataset_id: int) -> Sequence[EvalCase]:
        result = await self._session.scalars(
            select(EvalCaseORM)
            .where(EvalCaseORM.dataset_id == dataset_id)
            .order_by(EvalCaseORM.case_id.asc())
        )
        return [self._to_case(row) for row in result.all()]

    async def create_run(self, run: EvalRun) -> UUID:
        if run.id is None:
            raise ValueError("EvalRun.id must be set before persisting")

        obj = EvalRunORM(
            id=run.id,
            dataset_id=run.dataset_id,
            system_version=run.system_version,
            retriever_config_json=run.retriever_config_json,
            llm_config_json=run.llm_config_json,
        )
        self._session.add(obj)
        await self._session.flush()
        logger.info("Created eval run (id={}, dataset_id={})", obj.id, obj.dataset_id)
        return obj.id

    async def get_run(self, run_id: UUID) -> EvalRun | None:
        row = await self._session.scalar(
            select(EvalRunORM).where(EvalRunORM.id == run_id)
        )
        return self._to_run(row) if row is not None else None

    async def get_latest_run_id(self, dataset_id: int) -> UUID | None:
        return await self._session.scalar(
            select(EvalRunORM.id)
            .where(EvalRunORM.dataset_id == dataset_id)
            .order_by(EvalRunORM.created_at.desc())
            .limit(1)
        )

    async def upsert_result(self, result: EvalResult) -> None:
        obj = await self._session.get(
            EvalResultORM,
            {"eval_run_id": result.eval_run_id, "case_id": result.case_id},
        )
        if obj is None:
            obj = EvalResultORM(
                eval_run_id=result.eval_run_id,
                case_id=result.case_id,
                model_answer_text=result.model_answer_text,
                bert_score=result.bert_score,
                precision=result.precision,
                recall=result.recall,
                f1=result.f1,
                rouge_1=result.rouge_1,
                rouge_l=result.rouge_l,
                llm_judge_score=result.llm_judge_score,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                tokens_total=result.tokens_total,
            )
            self._session.add(obj)
            await self._session.flush()
            return

        obj.model_answer_text = result.model_answer_text
        obj.bert_score = result.bert_score
        obj.precision = result.precision
        obj.recall = result.recall
        obj.f1 = result.f1
        obj.rouge_1 = result.rouge_1
        obj.rouge_l = result.rouge_l
        obj.llm_judge_score = result.llm_judge_score
        obj.latency_ms = result.latency_ms
        obj.cost_usd = result.cost_usd
        obj.tokens_total = result.tokens_total
        await self._session.flush()

    async def list_results(self, run_id: UUID) -> Sequence[EvalResult]:
        result = await self._session.scalars(
            select(EvalResultORM)
            .where(EvalResultORM.eval_run_id == run_id)
            .order_by(EvalResultORM.case_id.asc())
        )
        return [self._to_result(row) for row in result.all()]

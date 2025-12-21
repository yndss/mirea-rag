from __future__ import annotations

import asyncio
import csv
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import UUID

from loguru import logger

from app.application.rag_service import RagAnswerDetails, RagService
from app.domain.models.eval import EvalCase, EvalResult, EvalRun
from app.eval.judge import LlmJudge
from app.eval.metrics import rouge_1_f1, rouge_l_f1, token_set_prf1
from app.infrastructure.config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_TIMEOUT,
    EMBEDDING_MODEL_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_TIMEOUT,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_MODEL_NAME,
    RAG_MIN_SIMILARITY,
    RAG_QA_PROMPT_NAME,
    RAG_TOP_K,
    SYSTEM_PROMPT_NAME,
)
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.db.eval_repository import SqlAlchemyEvalRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient

# from app.pricing import estimate_llm_cost_usd


@dataclass(frozen=True)
class EvalPipelineConfig:
    dataset_name: str
    dataset_description: str | None
    system_version: str

    rag_top_k: int = RAG_TOP_K
    rag_min_similarity: float = RAG_MIN_SIMILARITY
    rag_qa_prompt_name: str = RAG_QA_PROMPT_NAME

    answer_model_name: str = OPENROUTER_MODEL_NAME or ""
    answer_temperature: float = OPENROUTER_TEMPERATURE
    answer_system_prompt_name: str = SYSTEM_PROMPT_NAME
    answer_base_url: str = OPENROUTER_BASE_URL
    answer_timeout: float = OPENROUTER_TIMEOUT

    judge_model_name: str = ""
    judge_temperature: float = 0.0
    judge_system_prompt_name: str = "judge_system_prompt.md"
    judge_prompt_name: str = "judge_prompt.md"
    judge_base_url: str = OPENROUTER_BASE_URL
    judge_timeout: float = OPENROUTER_TIMEOUT

    metrics_embedding_model_name: str | None = None
    metrics_embedding_base_url: str = EMBEDDING_BASE_URL
    metrics_embedding_timeout: float = EMBEDDING_TIMEOUT

    concurrency: int = 3
    limit_cases: int | None = None


class EvalPipeline:

    def __init__(self) -> None:
        self._session_factory = SessionLocal

    async def load_dataset_from_csv(
        self,
        *,
        dataset_name: str,
        description: str | None,
        csv_path: str,
        replace_cases: bool = True,
    ) -> int:
        cases = _read_cases_csv(csv_path)

        async with self._session_factory() as session:
            repo = SqlAlchemyEvalRepository(session)
            dataset = await repo.get_or_create_dataset(dataset_name, description)
            if dataset.id is None:
                raise RuntimeError("Dataset id is not set after create")

            if replace_cases:
                eval_cases = [
                    EvalCase(
                        dataset_id=dataset.id,
                        case_id=idx,
                        question_text=q,
                        ideal_answer_text=a,
                        meta_json=None,
                    )
                    for idx, (q, a) in enumerate(cases, start=1)
                ]
                await repo.replace_cases(dataset.id, eval_cases)
            await session.commit()

            return int(dataset.id)

    async def run(self, config: EvalPipelineConfig) -> UUID:
        if not config.dataset_name:
            raise ValueError("dataset_name must be provided")
        if not config.system_version:
            raise ValueError("system_version must be provided")
        if not config.answer_model_name:
            raise ValueError("answer_model_name must be provided")
        if not config.judge_model_name:
            raise ValueError("judge_model_name must be provided (LLM-as-a-judge)")

        dataset_id, cases = await self._load_cases_for_run(
            dataset_name=config.dataset_name, limit_cases=config.limit_cases
        )
        if not cases:
            raise RuntimeError(
                f"No eval cases found for dataset '{config.dataset_name}'. "
                f"Load it first from CSV."
            )

        run_id = uuid.uuid4()
        await self._create_run(
            config=config,
            run=EvalRun(
                id=run_id,
                dataset_id=dataset_id,
                created_at=None,
                system_version=config.system_version,
                retriever_config_json={
                    "top_k": config.rag_top_k,
                    "min_similarity": config.rag_min_similarity,
                    "distance_metric": "cosine",
                    "embedding_model_name": EMBEDDING_MODEL_NAME,
                },
                llm_config_json={
                    "answer_model_name": config.answer_model_name,
                    "answer_temperature": config.answer_temperature,
                    "answer_system_prompt_name": config.answer_system_prompt_name,
                    "qa_prompt_name": config.rag_qa_prompt_name,
                    "judge_model_name": config.judge_model_name,
                    "judge_temperature": config.judge_temperature,
                    "judge_system_prompt_name": config.judge_system_prompt_name,
                    "judge_prompt_name": config.judge_prompt_name,
                    "metrics_embedding_model_name": config.metrics_embedding_model_name,
                },
            ),
        )

        answer_embedder = OpenRouterEmbeddingProvider()
        answer_llm = OpenRouterLlmClient(
            model_name=config.answer_model_name,
            temperature=config.answer_temperature,
            system_prompt_name=config.answer_system_prompt_name,
            base_url=config.answer_base_url,
            timeout=config.answer_timeout,
        )

        judge_llm = OpenRouterLlmClient(
            model_name=config.judge_model_name,
            temperature=config.judge_temperature,
            system_prompt_name=config.judge_system_prompt_name,
            base_url=config.judge_base_url,
            timeout=config.judge_timeout,
        )
        judge = LlmJudge(judge_llm, prompt_name=config.judge_prompt_name)

        metrics_embedder = (
            OpenRouterEmbeddingProvider(
                model_name=config.metrics_embedding_model_name,
                base_url=config.metrics_embedding_base_url,
                timeout=config.metrics_embedding_timeout,
            )
            if config.metrics_embedding_model_name
            else None
        )

        sem = asyncio.Semaphore(max(1, int(config.concurrency)))
        tasks = [
            asyncio.create_task(
                self._process_case(
                    sem=sem,
                    run_id=run_id,
                    case=case,
                    config=config,
                    answer_embedder=answer_embedder,
                    answer_llm=answer_llm,
                    judge=judge,
                    metrics_embedder=metrics_embedder,
                )
            )
            for case in cases
        ]

        await asyncio.gather(*tasks)

        await answer_embedder.close()
        await answer_llm.close()
        await judge_llm.close()
        if metrics_embedder is not None:
            await metrics_embedder.close()

        return run_id

    async def _load_cases_for_run(
        self, *, dataset_name: str, limit_cases: int | None
    ) -> tuple[int, Sequence[EvalCase]]:
        async with self._session_factory() as session:
            repo = SqlAlchemyEvalRepository(session)
            dataset = await repo.get_dataset_by_name(dataset_name)
            if dataset is None or dataset.id is None:
                return 0, []

            cases = await repo.list_cases(dataset.id)
            if limit_cases is not None:
                cases = list(cases)[: int(limit_cases)]
            return int(dataset.id), cases

    async def _create_run(self, *, config: EvalPipelineConfig, run: EvalRun) -> None:
        async with self._session_factory() as session:
            repo = SqlAlchemyEvalRepository(session)
            await repo.create_run(run)
            await session.commit()
            logger.info(
                "Eval run created (id={}, dataset={}, cases_limit={}, concurrency={})",
                run.id,
                config.dataset_name,
                config.limit_cases,
                config.concurrency,
            )

    async def _process_case(
        self,
        *,
        sem: asyncio.Semaphore,
        run_id: UUID,
        case: EvalCase,
        config: EvalPipelineConfig,
        answer_embedder: OpenRouterEmbeddingProvider,
        answer_llm: OpenRouterLlmClient,
        judge: LlmJudge,
        metrics_embedder: OpenRouterEmbeddingProvider | None,
    ) -> None:
        async with sem:
            answer_details: RagAnswerDetails | None = None
            answer_text: str = ""

            try:
                answer_details, answer_text = await self._answer_case(
                    case=case,
                    config=config,
                    answer_embedder=answer_embedder,
                    answer_llm=answer_llm,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to generate answer (case_id={}): {}", case.case_id, exc
                )
                await self._persist_result(
                    EvalResult(
                        eval_run_id=run_id,
                        case_id=case.case_id,
                        model_answer_text=f"ERROR: {exc}",
                        bert_score=None,
                        precision=None,
                        recall=None,
                        f1=None,
                        rouge_1=None,
                        rouge_l=None,
                        llm_judge_score=None,
                        latency_ms=None,
                        cost_usd=None,
                        tokens_total=None,
                    )
                )
                return

            overlap = token_set_prf1(case.ideal_answer_text, answer_text)
            rouge_1 = rouge_1_f1(case.ideal_answer_text, answer_text)
            rouge_l = rouge_l_f1(case.ideal_answer_text, answer_text)

            bert_score: float | None = None
            if metrics_embedder is not None:
                try:
                    bert_score = await _embedding_similarity(
                        metrics_embedder, case.ideal_answer_text, answer_text
                    )
                except Exception as exc:
                    logger.exception(
                        "Failed to compute bert_score (case_id={}): {}",
                        case.case_id,
                        exc,
                    )

            judge_score: int | None = None
            # judge_cost_usd: float | None = None
            # judge_tokens_total: int | None = None

            try:
                judged = await judge.judge(
                    question=case.question_text,
                    ideal_answer=case.ideal_answer_text,
                    model_answer=answer_text,
                )
                judge_score = judged.score
                # judge_tokens_total = (
                #     judged.generation.usage.total_tokens
                #     if judged.generation.usage
                #     else None
                # )
                # judge_cost_usd = estimate_llm_cost_usd(
                #     model_name=judged.generation.model or config.judge_model_name,
                #     prompt_tokens=(
                #         judged.generation.usage.prompt_tokens
                #         if judged.generation.usage
                #         else None
                #     ),
                #     completion_tokens=(
                #         judged.generation.usage.completion_tokens
                #         if judged.generation.usage
                #         else None
                #     ),
                # )
            except Exception as exc:
                logger.exception("Judge failed (case_id={}): {}", case.case_id, exc)

            total_cost_usd = answer_details.cost_usd
            tokens_total = answer_details.usage_total_tokens

            latency_ms = answer_details.latency_ms_total

            await self._persist_result(
                EvalResult(
                    eval_run_id=run_id,
                    case_id=case.case_id,
                    model_answer_text=answer_text,
                    bert_score=bert_score,
                    precision=overlap.precision,
                    recall=overlap.recall,
                    f1=overlap.f1,
                    rouge_1=rouge_1,
                    rouge_l=rouge_l,
                    llm_judge_score=judge_score,
                    latency_ms=latency_ms,
                    cost_usd=total_cost_usd,
                    tokens_total=tokens_total,
                )
            )

    async def _answer_case(
        self,
        *,
        case: EvalCase,
        config: EvalPipelineConfig,
        answer_embedder: OpenRouterEmbeddingProvider,
        answer_llm: OpenRouterLlmClient,
    ) -> tuple[RagAnswerDetails, str]:
        async with self._session_factory() as session:
            qa_repo = SqlAlchemyQaPairRepository(session)
            rag = RagService(
                qa_repo=qa_repo,
                embedding_provider=answer_embedder,
                llm_client=answer_llm,
                run_repo=None,
                top_k=config.rag_top_k,
                qa_prompt_name=config.rag_qa_prompt_name,
                min_similarity=config.rag_min_similarity,
            )
            details = await rag.answer_detailed(case.question_text, user_id=None)
            return details, details.answer_text

    async def _persist_result(self, result: EvalResult) -> None:
        async with self._session_factory() as session:
            repo = SqlAlchemyEvalRepository(session)
            await repo.upsert_result(result)
            await session.commit()


def _read_cases_csv(csv_path: str) -> list[tuple[str, str]]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cases: list[tuple[str, str]] = []
    for row in rows:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue
        cases.append((q, a))
    return cases


async def _embedding_similarity(
    embedder: OpenRouterEmbeddingProvider, a: str, b: str
) -> float:
    vecs = await embedder.embed_many([a, b])
    if len(vecs) != 2:
        raise RuntimeError("Unexpected embeddings count")
    va, vb = vecs[0], vecs[1]
    if len(va) != len(vb):
        raise RuntimeError("Embedding dimension mismatch")
    return float(sum(x * y for x, y in zip(va, vb, strict=False)))

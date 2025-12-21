import asyncio
import os

from loguru import logger

from app.eval.pipeline import EvalPipeline, EvalPipelineConfig
from app.eval.report import format_summary, summarize_run
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.eval_repository import SqlAlchemyEvalRepository
from app.infrastructure.logging import setup_logging


async def main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=os.getenv("EVAL_DATASET_NAME", "golden_set_v1")
    )
    parser.add_argument(
        "--system-version", default=os.getenv("SYSTEM_VERSION", "unknown")
    )
    parser.add_argument(
        "--concurrency", type=int, default=int(os.getenv("EVAL_CONCURRENCY", "3"))
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--answer-model",
        default=os.getenv("EVAL_ANSWER_MODEL_NAME")
        or os.getenv("OPENROUTER_MODEL_NAME", ""),
    )
    parser.add_argument(
        "--answer-temperature",
        type=float,
        default=float(
            os.getenv(
                "EVAL_ANSWER_TEMPERATURE", os.getenv("OPENROUTER_TEMPERATURE", "0.1")
            )
        ),
    )
    parser.add_argument(
        "--answer-system-prompt",
        default=os.getenv(
            "EVAL_ANSWER_SYSTEM_PROMPT_NAME",
            os.getenv("SYSTEM_PROMPT_NAME", "system_prompt.md"),
        ),
    )

    parser.add_argument("--judge-model", default=os.getenv("EVAL_JUDGE_MODEL_NAME", ""))
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--judge-system-prompt",
        default=os.getenv("EVAL_JUDGE_SYSTEM_PROMPT_NAME", "judge_system_prompt.md"),
    )
    parser.add_argument(
        "--judge-prompt", default=os.getenv("EVAL_JUDGE_PROMPT_NAME", "judge_prompt.md")
    )

    parser.add_argument(
        "--metrics-embedding-model",
        default=os.getenv("EVAL_METRICS_EMBEDDING_MODEL_NAME"),
    )
    args = parser.parse_args()

    pipeline = EvalPipeline()
    run_id = await pipeline.run(
        EvalPipelineConfig(
            dataset_name=args.dataset,
            dataset_description=None,
            system_version=args.system_version,
            answer_model_name=args.answer_model,
            answer_temperature=args.answer_temperature,
            answer_system_prompt_name=args.answer_system_prompt,
            judge_model_name=args.judge_model,
            judge_temperature=args.judge_temperature,
            judge_system_prompt_name=args.judge_system_prompt,
            judge_prompt_name=args.judge_prompt,
            metrics_embedding_model_name=args.metrics_embedding_model,
            concurrency=args.concurrency,
            limit_cases=args.limit,
        )
    )

    async with SessionLocal() as session:
        repo = SqlAlchemyEvalRepository(session)
        results = await repo.list_results(run_id)

    summary = summarize_run(run_id, results)
    print(format_summary(summary))

    logger.info("Eval run completed (id={})", run_id)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
from typing import Optional
from uuid import UUID

from app.eval.report import format_summary, summarize_run
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.eval_repository import SqlAlchemyEvalRepository
from app.infrastructure.logging import setup_logging


async def main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dataset", default=os.getenv("EVAL_DATASET_NAME", "golden_set_v1")
    )
    args = parser.parse_args()

    run_id: Optional[UUID] = UUID(args.run_id) if args.run_id else None

    async with SessionLocal() as session:
        repo = SqlAlchemyEvalRepository(session)
        if run_id is None:
            dataset = await repo.get_dataset_by_name(args.dataset)
            if dataset is None or dataset.id is None:
                raise RuntimeError(f"Unknown dataset: {args.dataset}")
            run_id = await repo.get_latest_run_id(dataset.id)
            if run_id is None:
                raise RuntimeError(f"No runs found for dataset: {args.dataset}")

        results = await repo.list_results(run_id)

    summary = summarize_run(run_id, results)
    print(format_summary(summary))


if __name__ == "__main__":
    asyncio.run(main())

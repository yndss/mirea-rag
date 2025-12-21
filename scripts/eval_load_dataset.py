import asyncio

from app.eval.pipeline import EvalPipeline
from app.infrastructure.logging import setup_logging


async def main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="golden_set_v1")
    parser.add_argument("--description", default=None)
    parser.add_argument("--csv", default="data/test.csv")
    parser.add_argument("--no-replace", action="store_true")
    args = parser.parse_args()

    pipeline = EvalPipeline()
    dataset_id = await pipeline.load_dataset_from_csv(
        dataset_name=args.dataset,
        description=args.description,
        csv_path=args.csv,
        replace_cases=not args.no_replace,
    )

    print(f"dataset_id={dataset_id}")


if __name__ == "__main__":
    asyncio.run(main())

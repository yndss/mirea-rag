import asyncio
import csv
from pathlib import Path

from loguru import logger

from app.domain.models.qa_pair import QaPair
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.logging import setup_logging


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    return default


async def seed_from_csv(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    async with SessionLocal() as session:
        repo = SqlAlchemyQaPairRepository(session)
        embedder = OpenRouterEmbeddingProvider()

        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            questions = [row["question"] for row in rows]
            logger.info(
                "Embedding questions from CSV (count={}, path={})", len(questions), path
            )
            embeddings = await embedder.embed_many(questions)

            qa_objects: list[QaPair] = []

            for row, emb in zip(rows, embeddings, strict=False):
                qa = QaPair(
                    id=None,
                    question=row["question"],
                    answer=row["answer"],
                    source_url=(row.get("source_url") or None),
                    topic=row["topic"],
                    is_generated=parse_bool(row.get("is_generated"), default=True),
                    embedding=emb,
                    created_at=None,
                )
                qa_objects.append(qa)

            await repo.add_many(qa_objects)
            await session.commit()
            logger.info("Inserted QA pairs into database (count={})", len(qa_objects))

        except Exception as exc:
            await session.rollback()
            logger.exception("Failed to seed QA pairs from CSV: {}", exc)
            raise


if __name__ == "__main__":
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to qa_pairs CSV file")
    args = parser.parse_args()

    asyncio.run(seed_from_csv(args.csv))

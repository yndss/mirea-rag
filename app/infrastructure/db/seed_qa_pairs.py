import csv
from pathlib import Path

from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.domain.models.qa_pair import QaPair


def parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    return default


def seed_from_csv(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    session = SessionLocal()
    repo = SqlAlchemyQaPairRepository(session)
    embedder = OpenRouterEmbeddingProvider()

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        questions = [row["question"] for row in rows]
        embeddings = embedder.embed_many(questions)

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

        repo.add_many(qa_objects)
        session.commit()
        print(f"Inserted {len(qa_objects)} QA pairs")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to qa_pairs CSV file")
    args = parser.parse_args()

    seed_from_csv(args.csv)

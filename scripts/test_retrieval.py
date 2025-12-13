import asyncio

from app.infrastructure.config import RAG_MAX_DISTANCE, RAG_TOP_K
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)


async def main() -> None:
    async with SessionLocal() as session:
        repo = SqlAlchemyQaPairRepository(session)
        embedder = OpenRouterEmbeddingProvider()

        user_question = "Сколько стоит обучение на платном отделении и есть ли рассрочка/оплата по семестрам?"

        query_vec = await embedder.embed(user_question)
        results = await repo.find_top_k(
            query_vec,
            k=RAG_TOP_K,
            max_distance=RAG_MAX_DISTANCE,
        )

        print(f"Запрос: {user_question}")
        print("Топ-5 найденных вопроса:\n")
        for i, qa in enumerate(results, start=1):
            print(f"{i}. {qa.question}")
            print(f"   topic={qa.topic}, is_generated={qa.is_generated}")
            print(f"   answer={qa.answer[:120]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())

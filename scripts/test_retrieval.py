import asyncio

from app.infrastructure.config import RAG_MIN_SIMILARITY, RAG_TOP_K
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
        hits = await repo.find_top_k(query_vec, k=RAG_TOP_K)
        used_hits = [hit for hit in hits if hit.similarity >= RAG_MIN_SIMILARITY]

        print(f"Запрос: {user_question}")
        print("Топ-5 найденных вопроса:\n")
        for hit in used_hits:
            qa = hit.qa_pair
            print(
                f"{hit.rank + 1}. id={qa.id}, similarity={hit.similarity:.4f}, distance={hit.distance:.4f}"
            )
            print(f"   question={qa.question}")
            print(f"   topic={qa.topic}, is_generated={qa.is_generated}")
            print(f"   answer={qa.answer[:120]}...")
            print()


if __name__ == "__main__":
    asyncio.run(main())

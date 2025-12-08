from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)


def main() -> None:
    session = SessionLocal()
    repo = SqlAlchemyQaPairRepository(session)
    embedder = OpenRouterEmbeddingProvider()

    try:
        user_question = "Сколько стоит обучение на платном отделении и есть ли рассрочка/оплата по семестрам?"

        query_vec = embedder.embed(user_question)
        results = repo.find_top_k(query_vec, k=5)

        print(f"Запрос: {user_question}")
        print("Топ-5 найденных вопроса:\n")
        for i, qa in enumerate(results, start=1):
            print(f"{i}. {qa.question}")
            print(f"   topic={qa.topic}, is_generated={qa.is_generated}")
            print(f"   answer={qa.answer[:120]}...")
            print()

    finally:
        session.close()


if __name__ == "__main__":
    main()

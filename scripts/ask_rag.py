from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient
from app.application.rag_service import RagService


def main() -> None:
    session = SessionLocal()
    try:
        qa_repo = SqlAlchemyQaPairRepository(session)
        embedding_provider = OpenRouterEmbeddingProvider()
        llm_client = OpenRouterLlmClient()

        rag_service = RagService(
            qa_repo=qa_repo,
            embedding_provider=embedding_provider,
            llm_client=llm_client,
            top_k=5,
        )

        print("RAG-консоль. Введи вопрос абитуриента. Пустая строка — выход.\n")

        while True:
            try:
                question = input("Вопрос > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                break

            if not question:
                print("Пустой ввод, выходим.")
                break

            print("\nДумаю...\n")
            try:
                answer = rag_service.answer(question)
            except Exception as e:
                print(f"Ошибка при обработке вопроса: {e}")
                continue

            print("Ответ:")
            print(answer)
            print("\n" + "-" * 60 + "\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()

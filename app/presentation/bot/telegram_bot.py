import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties

from app.infrastructure.config import TELEGRAM_BOT_TOKEN
from app.infrastructure.db.base import SessionLocal
from app.infrastructure.db.crud import SqlAlchemyQaPairRepository
from app.infrastructure.llm.openrouter_embedding_provider import (
    OpenRouterEmbeddingProvider,
)
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient
from app.application.rag_service import RagService


router = Router()


@asynccontextmanager
async def rag_service_context() -> AsyncIterator[RagService]:
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
        yield rag_service
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "Привет! Я бот-помощник абитуриентов МИРЭА 🎓\n\n"
        "Задавай вопросы про поступление, приёмную кампанию, направления, "
        "общежитие и другие вещи, связанные с МИРЭА.\n\n"
        "Например:\n"
        "• Какие документы нужны для поступления на бакалавриат?\n"
        "• Дают ли общежитие иногородним студентам?\n"
        "• Какие сроки подачи документов?\n\n"
        "Просто напиши свой вопрос одним сообщением"
    )
    await message.answer(text)


@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = (message.text or "").strip()
    if not question:
        await message.answer("Пожалуйста, напиши текстовый вопрос.")

    await message.chat.do("typing")

    try:
        async with rag_service_context() as rag_service:
            answer = await asyncio.to_thread(rag_service.answer, question)

    except Exception as e:
        await message.answer(
            "Произошла ошибка при обработке вопроса. " "Попробуй ещё раз позже."
        )
        print(f"Error while answering question: {e}")
        return

    await message.answer(answer)


async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)

    print("Telegram bot is starting polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

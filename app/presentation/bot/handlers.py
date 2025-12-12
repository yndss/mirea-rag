from aiogram import Router, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from loguru import logger

from app.infrastructure.config import TELEGRAM_WELCOME_PROMPT
from app.prompts import load_prompt
from .services import rag_service_context


router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    welcome_text = load_prompt(TELEGRAM_WELCOME_PROMPT)
    await message.answer(welcome_text)


@router.message(F.text)
async def handle_question(message: Message) -> None:
    question = (message.text or "").strip()
    if not question:
        await message.answer("Пожалуйста, напиши текстовый вопрос.")
        return

    logger.info(
        "Received question from user (chat_id={}, user_id={}, length={})",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        len(question),
    )
    await message.chat.do("typing")

    try:
        async with rag_service_context() as rag_service:
            answer = await rag_service.answer(question)
    except Exception as exc:
        await message.answer(
            "Произошла ошибка при обработке вопроса. Попробуй ещё раз позже."
        )
        logger.exception("Failed to process question: {}", exc)
        return

    logger.info(
        "Sending answer to user (chat_id={}, user_id={}, answer_len={})",
        message.chat.id,
        message.from_user.id if message.from_user else None,
        len(answer),
    )
    await message.answer(answer)

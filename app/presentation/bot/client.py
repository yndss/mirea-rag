import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from loguru import logger

from app.infrastructure.config import TELEGRAM_BOT_TOKEN
from app.infrastructure.logging import setup_logging
from .handlers import router


def _create_bot() -> Bot:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment")

    return Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def main() -> None:
    setup_logging()
    logger.info("Logging configured for Telegram bot")

    bot = _create_bot()
    dp = Dispatcher()
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)

    logger.info("Telegram bot is starting polling")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

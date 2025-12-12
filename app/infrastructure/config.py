import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mirea_rag")
DB_USER = os.getenv("DB_USER", "mirea_user")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "postgresql+asyncpg")


def build_db_url() -> str:
    return f"{DB_DRIVER}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


DB_URL = build_db_url()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "60.0"))
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.1"))
SYSTEM_PROMPT_NAME = os.getenv("SYSTEM_PROMPT_NAME", "system_prompt.md")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENROUTER_BASE_URL)
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "30.0"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_QA_PROMPT_NAME = os.getenv("RAG_QA_PROMPT_NAME", "qa_prompt.md")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WELCOME_PROMPT = os.getenv("TELEGRAM_WELCOME_PROMPT", "telegram_welcome.md")

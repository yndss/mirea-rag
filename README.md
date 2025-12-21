# mirea-rag

RAG‑проект для вопросов абитуриентов МИРЭА. Использует Postgres + pgvector для хранения, OpenRouter для эмбеддингов/LLM. Можно запускать как Telegram‑бота или из консоли.

## Быстрый старт (локально)

1. Скопируй `.env.example` в `.env` и заполни обязательные переменные:
   - `DB_*` для Postgres
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL_NAME`
   - `EMBEDDING_MODEL_NAME`
   - `TELEGRAM_BOT_TOKEN` (нужен только для бота)
2. Подними Postgres с pgvector:
   - `docker-compose up -d db`
3. Установи зависимости и применяй миграции:
   - `python -m venv .venv`
   - `. .venv/bin/activate`
   - `pip install -r requirements.txt`
   - `alembic upgrade head`

## Загрузка данных

Читает CSV с вопросами/ответами и сохраняет эмбеддинги по каждой строке.

`python -m app.infrastructure.db.seed_qa_pairs --csv data/qa_pairs.csv`

## Запуск бота

`python -m app.presentation.bot.client`

## Консольный режим

Интерактивный ввод вопросов без Telegram.

`python -m scripts.ask_rag`

## Проверка ретривера

Печатает топ‑совпадения для примера вопроса.

`python -m scripts.test_retrieval`

## Eval‑пайплайн

1. Загрузить датасет из CSV:
   - `python -m scripts.eval_load_dataset --dataset golden_set_v1 --csv data/test.csv`
2. Запустить eval:
   - `python -m scripts.eval_run`
   - Опционально есть другие параметры
3. Показать отчёт по последнему запуску:
   - `python -m scripts.eval_report`
   - Или по id: `python -m scripts.eval_report --run-id <uuid>`
   - Или по названию датасета: `python -m scripts.eval_report --dataset golden_set_v1`

## Docker

Полный стек (db + app):

`docker-compose up --build`

Контейнер сам прогоняет миграции и запускает Telegram‑бота.

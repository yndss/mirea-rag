from loguru import logger
from openai import AsyncOpenAI

from app.infrastructure.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL_NAME,
    OPENROUTER_TEMPERATURE,
    OPENROUTER_TIMEOUT,
    SYSTEM_PROMPT_NAME,
)
from app.prompts import load_prompt


class OpenRouterLlmClient:

    def __init__(self, system_prompt_name: str = SYSTEM_PROMPT_NAME) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not OPENROUTER_MODEL_NAME:
            raise RuntimeError("OPENROUTER_MODEL_NAME is not set")

        self._model = OPENROUTER_MODEL_NAME
        self._system_prompt = load_prompt(system_prompt_name)
        self._client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            timeout=OPENROUTER_TIMEOUT,
        )

    async def generate(self, prompt: str) -> str:
        logger.info(
            "Sending prompt to OpenRouter (model={})",
            self._model,
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=OPENROUTER_TEMPERATURE,
            )
            answer = response.choices[0].message.content or ""
            logger.debug(
                "OpenRouter responded (finish_reason={}, answer_len={})",
                response.choices[0].finish_reason,
                len(answer),
            )
            return answer
        except Exception as exc:
            logger.exception("OpenRouter completion request failed: {}", exc)
            raise

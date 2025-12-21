from loguru import logger
from openai import AsyncOpenAI

from app.domain.models.llm_generation import LlmGeneration, LlmUsage
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

    def __init__(
        self,
        *,
        model_name: str | None = None,
        temperature: float | None = None,
        system_prompt_name: str = SYSTEM_PROMPT_NAME,
        base_url: str = OPENROUTER_BASE_URL,
        api_key: str | None = OPENROUTER_API_KEY,
        timeout: float = OPENROUTER_TIMEOUT,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        resolved_model_name = model_name or OPENROUTER_MODEL_NAME
        if not resolved_model_name:
            raise RuntimeError("OPENROUTER_MODEL_NAME is not set")

        self._model = resolved_model_name
        self._temperature = (
            OPENROUTER_TEMPERATURE if temperature is None else float(temperature)
        )
        self._system_prompt = load_prompt(system_prompt_name)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    async def generate(self, prompt: str) -> LlmGeneration:
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
                temperature=self._temperature,
            )
            answer = response.choices[0].message.content or ""
            usage = (
                LlmUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else None
            )
            logger.debug(
                "OpenRouter responded (finish_reason={}, answer_len={})",
                response.choices[0].finish_reason,
                len(answer),
            )
            return LlmGeneration(
                text=answer,
                model=response.model,
                usage=usage,
            )
        except Exception as exc:
            logger.exception("OpenRouter completion request failed: {}", exc)
            raise

    async def close(self) -> None:
        await self._client.close()

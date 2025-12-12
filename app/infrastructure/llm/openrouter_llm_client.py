from loguru import logger
from openai import AsyncOpenAI

from app.infrastructure.config import OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME
from app.prompts import load_prompt


class OpenRouterLlmClient:

    def __init__(
        self,
        timeout: float = 60.0,
        base_url: str = "https://openrouter.ai/api/v1",
        system_prompt_name: str = "system_prompt.md",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not OPENROUTER_MODEL_NAME:
            raise RuntimeError("OPENROUTER_MODEL_NAME is not set")

        self._model = OPENROUTER_MODEL_NAME
        self._system_prompt = load_prompt(system_prompt_name)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=OPENROUTER_API_KEY,
            default_headers=extra_headers,
            timeout=timeout,
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
                temperature=0.1,
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

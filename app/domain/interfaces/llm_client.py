from typing import Protocol

from app.domain.models.llm_generation import LlmGeneration


class LlmClient(Protocol):
    async def generate(self, prompt: str) -> LlmGeneration: ...

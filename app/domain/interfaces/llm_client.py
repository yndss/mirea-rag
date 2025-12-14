from typing import Protocol


class LlmClient(Protocol):
    async def generate(self, prompt: str) -> str: ...

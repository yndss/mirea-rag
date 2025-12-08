from typing import Protocol


class LlmClient(Protocol):
    def generate(self, prompt: str) -> str: ...

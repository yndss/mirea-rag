from dataclasses import dataclass


@dataclass(frozen=True)
class LlmUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LlmGeneration:
    text: str
    model: str | None
    usage: LlmUsage | None

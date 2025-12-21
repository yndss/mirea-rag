from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LlmUsage:
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True)
class LlmGeneration:
    text: str
    model: Optional[str]
    usage: Optional[LlmUsage]

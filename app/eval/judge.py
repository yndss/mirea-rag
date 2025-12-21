from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from app.domain.models.llm_generation import LlmGeneration
from app.infrastructure.llm.openrouter_llm_client import OpenRouterLlmClient
from app.prompts.loader import load_prompt


@dataclass(frozen=True)
class JudgeResult:
    score: Optional[int]
    reason: Optional[str]
    generation: LlmGeneration
    raw_text: str


class LlmJudge:

    def __init__(self, llm_client: OpenRouterLlmClient, prompt_name: str) -> None:
        self._llm = llm_client
        self._template = load_prompt(prompt_name)

    def _build_prompt(
        self, *, question: str, ideal_answer: str, model_answer: str
    ) -> str:
        return (
            self._template.replace("{{question}}", question)
            .replace("{{ideal_answer}}", ideal_answer)
            .replace("{{model_answer}}", model_answer)
        )

    async def judge(
        self, *, question: str, ideal_answer: str, model_answer: str
    ) -> JudgeResult:
        prompt = self._build_prompt(
            question=question,
            ideal_answer=ideal_answer,
            model_answer=model_answer,
        )
        generation = await self._llm.generate(prompt)

        score, reason = _parse_judge_json(generation.text)
        if score is None:
            logger.warning("Judge returned unparsable score, raw={}", generation.text)

        return JudgeResult(
            score=score,
            reason=reason,
            generation=generation,
            raw_text=generation.text,
        )


def _parse_judge_json(text: str) -> tuple[Optional[int], Optional[str]]:
    obj_text = _extract_first_json_object(text)
    if obj_text is None:
        return None, None

    try:
        data = json.loads(obj_text)
    except Exception:
        return None, None

    score_raw = data.get("score")
    try:
        score = int(float(score_raw))
    except Exception:
        score = None

    if score is not None and not (1 <= score <= 5):
        score = None

    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        reason = str(reason)

    return score, reason


def _extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

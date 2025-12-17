import json
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from loguru import logger

_PRICING_PATH = Path(__file__).resolve().with_name("model_pricing.json")


class ModelPricing(TypedDict):
    in_per_1m: float
    out_per_1m: float


@lru_cache(maxsize=1)
def load_model_pricing() -> dict[str, ModelPricing]:
    raw = json.loads(_PRICING_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid pricing JSON format: {_PRICING_PATH}")
    return raw


def estimate_llm_cost_usd(
    *,
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    if not model_name:
        return None
    if prompt_tokens is None or completion_tokens is None:
        return None

    pricing = load_model_pricing().get(model_name)
    if pricing is None:
        logger.debug("No pricing configured for model: {}", model_name)
        return None

    in_per_1m = Decimal(str(pricing["in_per_1m"]))
    out_per_1m = Decimal(str(pricing["out_per_1m"]))
    cost = (
        (Decimal(prompt_tokens) * in_per_1m) + (Decimal(completion_tokens) * out_per_1m)
    ) / Decimal(1_000_000)

    return float(cost.quantize(Decimal("0.0000000001"), rounding=ROUND_HALF_UP))


__all__ = ["estimate_llm_cost_usd", "load_model_pricing"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
from uuid import UUID

from app.domain.models.eval import EvalResult


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: Optional[float]
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]


@dataclass(frozen=True)
class EvalSummary:
    eval_run_id: UUID
    cases_total: int
    cases_scored: int
    success_rate_ge_4: Optional[float]
    judge: Distribution
    bert_score_mean: Optional[float]
    rouge_1_mean: Optional[float]
    rouge_l_mean: Optional[float]
    latency_ms_mean: Optional[float]
    cost_usd_mean: Optional[float]
    tokens_total_mean: Optional[float]
    judge_cost_usd_mean: Optional[float]
    judge_tokens_total_mean: Optional[float]


def summarize_run(eval_run_id: UUID, results: Sequence[EvalResult]) -> EvalSummary:
    judge_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
    judge_dist = _distribution([float(x) for x in judge_scores])

    success_rate_ge_4 = (
        (sum(1 for x in judge_scores if x >= 4) / len(judge_scores) * 100.0)
        if judge_scores
        else None
    )

    return EvalSummary(
        eval_run_id=eval_run_id,
        cases_total=len(results),
        cases_scored=len(judge_scores),
        success_rate_ge_4=success_rate_ge_4,
        judge=judge_dist,
        bert_score_mean=_mean(r.bert_score for r in results),
        rouge_1_mean=_mean(r.rouge_1 for r in results),
        rouge_l_mean=_mean(r.rouge_l for r in results),
        latency_ms_mean=_mean(r.latency_ms for r in results),
        cost_usd_mean=_mean(r.cost_usd for r in results),
        tokens_total_mean=_mean(r.tokens_total for r in results),
        judge_cost_usd_mean=_mean(r.judge_cost_usd for r in results),
        judge_tokens_total_mean=_mean(r.judge_tokens_total for r in results),
    )


def format_summary(summary: EvalSummary) -> str:
    lines: list[str] = []
    lines.append(f"eval_run_id={summary.eval_run_id}")
    lines.append(f"cases_total={summary.cases_total}")
    lines.append(f"cases_scored={summary.cases_scored}")
    if summary.success_rate_ge_4 is not None:
        lines.append(f"success_rate_ge_4={summary.success_rate_ge_4:.1f}%")

    if summary.judge.count:
        lines.append(
            "judge_score: "
            f"mean={_fmt(summary.judge.mean)} "
            f"p10={_fmt(summary.judge.p10)} "
            f"p50={_fmt(summary.judge.p50)} "
            f"p90={_fmt(summary.judge.p90)}"
        )

    lines.append(f"bert_score_mean={_fmt(summary.bert_score_mean)}")
    lines.append(f"rouge_1_mean={_fmt(summary.rouge_1_mean)}")
    lines.append(f"rouge_l_mean={_fmt(summary.rouge_l_mean)}")
    lines.append(f"latency_ms_mean={_fmt(summary.latency_ms_mean)}")
    lines.append(f"cost_usd_mean={_fmt(summary.cost_usd_mean)}")
    lines.append(f"tokens_total_mean={_fmt(summary.tokens_total_mean)}")
    lines.append(f"judge_cost_usd_mean={_fmt(summary.judge_cost_usd_mean)}")
    lines.append(f"judge_tokens_total_mean={_fmt(summary.judge_tokens_total_mean)}")

    return "\n".join(lines)


def _mean(values: Iterable[Optional[float | int]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _distribution(values: Sequence[float]) -> Distribution:
    if not values:
        return Distribution(count=0, mean=None, p10=None, p50=None, p90=None)

    sorted_vals = sorted(values)
    return Distribution(
        count=len(sorted_vals),
        mean=sum(sorted_vals) / len(sorted_vals),
        p10=_percentile(sorted_vals, 10),
        p50=_percentile(sorted_vals, 50),
        p90=_percentile(sorted_vals, 90),
    )


def _percentile(sorted_values: Sequence[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])

    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "null"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.5f}"

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class EvalDataset:
    id: int | None
    name: str
    description: str | None
    created_at: datetime | None


@dataclass
class EvalCase:
    dataset_id: int
    case_id: int
    question_text: str
    ideal_answer_text: str
    meta_json: dict[str, Any] | None


@dataclass
class EvalRun:
    id: UUID | None
    dataset_id: int
    created_at: datetime | None
    system_version: str
    retriever_config_json: dict[str, Any] | None
    llm_config_json: dict[str, Any] | None


@dataclass
class EvalResult:
    eval_run_id: UUID
    case_id: int
    model_answer_text: str
    bert_score: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    rouge_1: float | None
    rouge_l: float | None
    llm_judge_score: int | None
    latency_ms: int | None
    cost_usd: float | None
    tokens_total: int | None

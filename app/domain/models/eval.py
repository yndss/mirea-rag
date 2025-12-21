from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID


@dataclass
class EvalDataset:
    id: Optional[int]
    name: str
    description: Optional[str]
    created_at: Optional[datetime]


@dataclass
class EvalCase:
    dataset_id: int
    case_id: int
    question_text: str
    ideal_answer_text: str
    meta_json: Optional[dict[str, Any]]


@dataclass
class EvalRun:
    id: Optional[UUID]
    dataset_id: int
    created_at: Optional[datetime]
    system_version: str
    retriever_config_json: Optional[dict[str, Any]]
    llm_config_json: Optional[dict[str, Any]]


@dataclass
class EvalResult:
    eval_run_id: UUID
    case_id: int
    model_answer_text: str
    bert_score: Optional[float]
    rouge_1: Optional[float]
    rouge_l: Optional[float]
    llm_judge_score: Optional[int]
    latency_ms: Optional[int]
    cost_usd: Optional[float]
    tokens_total: Optional[int]
    judge_cost_usd: Optional[float]
    judge_tokens_total: Optional[int]

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class RagRun:
    id: UUID | None
    created_at: datetime | None
    user_id: int | None
    question_text: str
    retriever_top_k: int
    similarity_threshold: float
    distance_metric: str
    context_text: str
    final_prompt_text: str
    model_name: str
    temperature: float
    extra_params: dict[str, Any] | None
    answer_text: str
    usage_prompt_tokens: int | None
    usage_completion_tokens: int | None
    usage_total_tokens: int | None
    cost_usd: float | None
    latency_ms_total: int | None
    latency_ms_retrieval: int | None
    latency_ms_llm: int | None
    latency_ms_embedding: int | None


@dataclass
class RagRunHit:
    rag_run_id: UUID | None
    rank: int
    qa_pair_id: int | None
    distance: float
    similarity: float
    used_in_context: bool

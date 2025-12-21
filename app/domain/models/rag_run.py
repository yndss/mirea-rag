from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID


@dataclass
class RagRun:
    id: Optional[UUID]
    created_at: Optional[datetime]
    user_id: Optional[int]
    question_text: str
    retriever_top_k: int
    similarity_threshold: float
    distance_metric: str
    context_text: str
    final_prompt_text: str
    model_name: str
    temperature: float
    extra_params: Optional[dict[str, Any]]
    answer_text: str
    usage_prompt_tokens: Optional[int]
    usage_completion_tokens: Optional[int]
    usage_total_tokens: Optional[int]
    cost_usd: Optional[float]
    latency_ms_total: Optional[int]
    latency_ms_retrieval: Optional[int]
    latency_ms_llm: Optional[int]
    latency_ms_embedding: Optional[int]


@dataclass
class RagRunHit:
    rag_run_id: Optional[UUID]
    rank: int
    qa_pair_id: Optional[int]
    distance: float
    similarity: float
    used_in_context: bool

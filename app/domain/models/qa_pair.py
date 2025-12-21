from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence


@dataclass
class QaPair:
    id: Optional[int]
    question: str
    answer: str
    source_url: Optional[str]
    topic: str
    is_generated: bool
    embedding: Sequence[float]
    created_at: Optional[datetime] = None


@dataclass
class QaPairHit:
    qa_pair: QaPair
    rank: int
    distance: float
    similarity: float

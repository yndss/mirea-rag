from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass
class QaPair:
    id: int | None
    question: str
    answer: str
    source_url: str | None
    topic: str
    is_generated: bool
    embedding: Sequence[float]
    created_at: datetime | None = None

"""Data models for Search Orchestrator and Client Integration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateDocument:
    """Document candidate passed to the reranker for cross-encoder scoring."""

    id: str
    passage: str
    initial_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    """Reranker score and final ranking output."""

    id: str
    score: float
    rank: int
    fallback_applied: bool = False
    fallback_reason: str | None = None

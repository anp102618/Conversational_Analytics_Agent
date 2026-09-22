"""Data schema for EDA RAG retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    """Represents one retrieved EDA chunk."""

    chunk_id: str
    content: str
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the retrieval result."""

        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty.")

        if not self.content.strip():
            raise ValueError("Retrieved content cannot be empty.")

        if self.rank < 0:
            raise ValueError("rank cannot be negative.")

@dataclass(slots=True)
class RetrievalResponse:
    """Represents the output of the RAG retrieval pipeline."""

    query: str
    dataset_id: str
    results: list[RetrievalResult] = field(default_factory=list)
    retrieval_top_k: int = 0
    rerank_top_k: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

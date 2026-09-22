"""Data schema for EDA document chunks used by the RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EDAChunk:
    """Represents a semantically meaningful EDA document chunk."""

    chunk_id: str
    document_id: str
    dataset_id: str
    analysis_type: str
    content: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the EDA chunk."""

        fields = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "analysis_type": self.analysis_type,
        }

        for name, value in fields.items():
            if not value.strip():
                raise ValueError(f"{name} cannot be empty.")

        if not self.content.strip():
            raise ValueError("EDA chunk content cannot be empty.")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")

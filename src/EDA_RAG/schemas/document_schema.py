"""Data schema for EDA source documents used by the RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EDADocument:
    """Represents a complete EDA summary document."""

    document_id: str
    dataset_id: str
    analysis_type: str
    source_path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the EDA document."""

        fields = {
            "document_id": self.document_id,
            "dataset_id": self.dataset_id,
            "analysis_type": self.analysis_type,
            "source_path": self.source_path,
        }

        for name, value in fields.items():
            if not value.strip():
                raise ValueError(f"{name} cannot be empty.")

        if not self.content.strip():
            raise ValueError("EDA document content cannot be empty.")

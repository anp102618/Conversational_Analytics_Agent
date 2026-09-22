"""Schema for parsed EDA sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EDASection:
    """Logical analytical section extracted from an EDA document."""

    section_id: str
    document_id: str
    dataset_id: str
    analysis_type: str
    title: str
    content: str
    section_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the EDA section."""
        for name in (
            "section_id",
            "document_id",
            "dataset_id",
            "analysis_type",
            "title",
            "content",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty.")

        if self.section_index < 0:
            raise ValueError("section_index cannot be negative.")

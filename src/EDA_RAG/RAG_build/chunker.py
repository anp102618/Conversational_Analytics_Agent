"""
Semantic chunker for EDA RAG.

Converts parsed EDASection objects into retrieval-ready EDAChunk
objects.

Chunking strategy
-----------------
1. Preserve the complete analytical section whenever possible.
2. Prefer semantic boundaries over arbitrary character boundaries.
3. Split oversized sections only when necessary.
4. Preserve metadata on every generated chunk.
5. Add lightweight contextual prefixes to chunks so that
   embeddings retain analytical context.

Examples
--------
Univariate
    Feature: tenure
    -> statistics remain associated with tenure.

Bivariate
    tenure vs MonthlyCharges
    -> correlation/effect-size evidence remains associated
       with the feature pair.

Multivariate
    Random Forest
    -> feature importance evidence remains associated
       with the method.
"""
from __future__ import annotations

import re
from typing import Any

from src.EDA_RAG.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)
from src.EDA_RAG.schemas.chunk_schema import EDAChunk
from src.EDA_RAG.schemas.section_schema import EDASection
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class EDASemanticChunker:
    """
    Convert EDA sections into semantic retrieval chunks.

    Parameters
    ----------
    chunk_size:
        Maximum preferred chunk size in characters.
    chunk_overlap:
        Number of characters retained between fallback
        subchunks when a section is too large.
    min_chunk_size:
        Minimum useful chunk size.

    Notes
    -----
    ``chunk_size`` is a fallback constraint, not the primary
    semantic boundary. EDA sections are kept intact whenever
    they fit within the configured size.
    """

    ANALYSIS_LABELS: dict[str, str] = {
        "univariate": "Univariate EDA",
        "bivariate": "Bivariate EDA",
        "multivariate": "Multivariate EDA",
    }

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ) -> None:
        """Initialize the semantic EDA chunker."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be greater than zero.")
        if min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size cannot exceed chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = get_log("EDASemanticChunker")
        self.logger.info(
            "Initialized EDA semantic chunker: chunk_size=%d, overlap=%d, min_size=%d",
            self.chunk_size,
            self.chunk_overlap,
            self.min_chunk_size,
        )

    # Public API
    @track_performance
    def chunk_section(self, section: EDASection) -> list[EDAChunk]:
        """
        Convert one EDA section into semantic chunks.

        Parameters
        ----------
        section:
            Parsed EDA section.

        Returns
        -------
        list[EDAChunk]
            One or more retrieval-ready chunks.
        """
        try:
            if not isinstance(section, EDASection):
                raise TypeError("section must be an EDASection.")

            self.logger.debug(
                "Chunking section_id=%s, title=%s",
                section.section_id,
                section.title,
            )

            contextual_content = self._build_contextual_content(section)
            if len(contextual_content) <= self.chunk_size:
                return [
                    self._create_chunk(
                        section=section,
                        content=contextual_content,
                        chunk_index=0,
                        total_chunks=1,
                    )
                ]

            subchunks = self._split_large_section(contextual_content)
            chunks = [
                self._create_chunk(
                    section=section,
                    content=content,
                    chunk_index=index,
                    total_chunks=len(subchunks),
                )
                for index, content in enumerate(subchunks)
            ]

            self.logger.debug(
                "Created %d chunks from section_id=%s",
                len(chunks),
                section.section_id,
            )
            return chunks
        except Exception as exc:
            self.logger.error(
                "Failed to chunk section_id=%s: %s",
                getattr(section, "section_id", "unknown"),
                exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def chunk_sections(self, sections: list[EDASection]) -> list[EDAChunk]:
        """
        Convert all parsed sections into chunks.

        Parameters
        ----------
        sections:
            Parsed EDA sections.

        Returns
        -------
        list[EDAChunk]
            Retrieval-ready EDA chunks.
        """
        try:
            if not sections:
                self.logger.warning("No EDA sections supplied for chunking.")
                return []

            chunks: list[EDAChunk] = []
            for section in sections:
                chunks.extend(self.chunk_section(section))

            chunks = self._finalize_chunk_metadata(chunks)
            self.logger.info(
                "Created %d chunks from %d EDA sections.",
                len(chunks),
                len(sections),
            )
            return chunks
        except Exception as exc:
            self.logger.error("Failed to chunk EDA sections: %s", exc)
            raise CustomException(exc, self.logger) from exc

    # Context construction
    @track_performance
    def _build_contextual_content(self, section: EDASection) -> str:
        """
        Add analytical context before the section content.

        The contextual prefix is intentionally short.

        Example
        -------
        [Analysis: Univariate EDA]
        [Feature: tenure]
        Feature: tenure
        Mean: 32.37
        Median: 29.00
        ...

        This improves embedding retrieval because a chunk
        containing only statistical values still carries the
        identity of the feature and analysis type.
        """
        analysis_type = section.analysis_type.lower().strip()
        analysis_label = self.ANALYSIS_LABELS.get(
            analysis_type, section.analysis_type.title()
        )
        prefix_lines = [f"[Analysis: {analysis_label}]"]

        metadata = section.metadata
        feature = metadata.get("feature")
        feature_1 = metadata.get("feature_1")
        feature_2 = metadata.get("feature_2")
        feature_pair = metadata.get("feature_pair")
        subtype = metadata.get("subtype")
        method = metadata.get("method")

        if feature:
            prefix_lines.append(f"[Feature: {feature}]")
        if feature_pair:
            prefix_lines.append(f"[Feature Pair: {feature_pair}]")
        elif feature_1 and feature_2:
            prefix_lines.append(f"[Feature Pair: {feature_1} vs {feature_2}]")
        if subtype:
            prefix_lines.append(f"[Bivariate Type: {subtype}]")
        if method:
            prefix_lines.append(f"[Method: {method}]")

        prefix_lines.append(f"[Section: {section.title}]")
        prefix = "\n".join(prefix_lines)
        return f"{prefix}\n\n{section.content.strip()}".strip()

    # Large-section splitting
    @track_performance
    def _split_large_section(self, content: str) -> list[str]:
        """
        Split an oversized section while preserving semantic
        boundaries where possible.

        Splitting priority
        ------------------
        1. Blank-line paragraphs
        2. Newline boundaries
        3. Sentence boundaries
        4. Character-level fallback

        The method does not blindly split at ``chunk_size``.
        """
        paragraphs = self._split_paragraphs(content)
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if not current:
                current = paragraph
                continue
            candidate = f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            chunks.append(current.strip())
            current = paragraph

        if current.strip():
            chunks.append(current.strip())

        # A paragraph itself may exceed chunk_size.
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._split_long_text(chunk))

        return self._apply_overlap(final_chunks)

    @staticmethod
    @track_performance
    def _split_paragraphs(content: str) -> list[str]:
        """
        Split content on blank-line boundaries.
        """
        return [
            p.strip()
            for p in re.split(r"\n\s*\n+", content)
            if p.strip()
        ]

    @track_performance
    def _split_long_text(self, text: str) -> list[str]:
        """
        Split a long text block using increasingly smaller
        semantic boundaries.
        """
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # First attempt: lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks: list[str] = []
        current = ""
        for line in lines:
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            if len(line) <= self.chunk_size:
                current = line
            else:
                chunks.extend(self._split_by_sentence_or_chars(line))
                current = ""

        if current:
            chunks.append(current.strip())
        return chunks

    @track_performance
    def _split_by_sentence_or_chars(self, text: str) -> list[str]:
        """
        Split very long text by sentence boundaries and,
        as a final fallback, character boundaries.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}" if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current.strip())
            if len(sentence) <= self.chunk_size:
                current = sentence
            else:
                chunks.extend(self._hard_split(sentence))
                current = ""

        if current:
            chunks.append(current.strip())
        return chunks

    @track_performance
    def _hard_split(self, text: str) -> list[str]:
        """
        Final character-level fallback.

        Used only when a single line/sentence is larger
        than the configured chunk size.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks

    # Overlap
    @track_performance
    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """
        Apply lightweight overlap between fallback chunks.

        Overlap is used only after semantic splitting.
        The overlap is taken from the end of the previous
        chunk and prepended to the next chunk.
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for index in range(1, len(chunks)):
            previous = chunks[index - 1]
            current = chunks[index]
            overlap_text = self._get_overlap_text(previous)
            combined = f"{overlap_text}\n{current}".strip()
            # Never allow overlap to make the chunk larger than the configured size.
            if len(combined) > self.chunk_size:
                combined = current
            result.append(combined)
        return result

    @track_performance
    def _get_overlap_text(self, text: str) -> str:
        """
        Extract an overlap from the end of a chunk.

        The overlap attempts to preserve complete lines
        instead of cutting through a statistic.
        """
        if len(text) <= self.chunk_overlap:
            return text
        candidate = text[-self.chunk_overlap:]
        newline_position = candidate.find("\n")
        if newline_position >= 0:
            candidate = candidate[newline_position + 1:]
        return candidate.strip()

    # Chunk construction
    @track_performance
    def _create_chunk(
        self,
        section: EDASection,
        content: str,
        chunk_index: int,
        total_chunks: int,
    ) -> EDAChunk:
        """
        Create one EDAChunk from a section.
        """
        metadata = dict(section.metadata)
        metadata.update({
            "section_id": section.section_id,
            "section_title": section.title,
            "source": section.metadata.get("source"),
            "chunk_index": chunk_index,
            "total_section_chunks": total_chunks,
            "chunking_strategy": "semantic_eda",
        })
        return EDAChunk(
            chunk_id=f"{section.section_id}-CHUNK-{chunk_index:03d}",
            document_id=section.document_id,
            dataset_id=section.dataset_id,
            analysis_type=section.analysis_type,
            content=content.strip(),
            chunk_index=chunk_index,
            metadata=metadata,
        )

    @staticmethod
    @track_performance
    def _finalize_chunk_metadata(chunks: list[EDAChunk]) -> list[EDAChunk]:
        """
        Add global chunk information.

        This preserves local section indexes while also
        providing a deterministic global position.
        """
        total_chunks = len(chunks)
        for global_index, chunk in enumerate(chunks):
            chunk.metadata["global_chunk_index"] = global_index
            chunk.metadata["total_chunks"] = total_chunks
        return chunks
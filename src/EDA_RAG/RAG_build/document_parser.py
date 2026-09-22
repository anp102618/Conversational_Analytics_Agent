"""
Deterministic parser for generated EDA summary documents.

The parser converts complete TXT summaries into logical analytical
sections suitable for semantic chunking and RAG indexing.

Supported analytical structures
--------------------------------

Univariate
    Feature -> all associated statistics

Bivariate
    Analysis subtype -> feature pair -> associated statistics

Multivariate
    Analytical method -> associated evidence

The parser intentionally does not:
    - calculate statistics
    - modify analytical values
    - call an LLM
    - generate embeddings
    - perform retrieval
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.EDA_RAG.schemas.document_schema import EDADocument
from src.EDA_RAG.schemas.section_schema import EDASection
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


@dataclass(slots=True)
class _SectionBuffer:
    """
    Internal representation of a section being accumulated.
    """
    title: str
    lines: list[str]
    metadata: dict[str, Any]


class EDADocumentParser:
    """
    Parse generated EDA summary documents into logical sections.

    The parser uses different structural rules for:

    - univariate EDA
    - bivariate EDA
    - multivariate EDA

    This is preferable to one generic parser because each EDA
    family has a different natural retrieval boundary.
    """

    # Generic patterns
    FEATURE_PATTERN = re.compile(
        r"^\s*(?:feature|variable|column)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    PAIR_PATTERN = re.compile(
        r"^\s*(?:feature\s*)?(?:pair|relationship)\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    SUBTYPE_PATTERN = re.compile(
        r"^\s*(?:analysis\s*)?(num_num|num_cat|cat_cat|num_datetime|cat_datetime)\s*$",
        re.IGNORECASE,
    )
    SECTION_LINE_PATTERN = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")

    # Known univariate sections
    UNIVARIATE_SUBSECTIONS = {
        "basic statistics", "central tendency", "dispersion statistics",
        "dispersion", "quantiles", "distribution shape", "missing values",
        "zero values", "negative values", "unique values", "outlier statistics",
        "outlier bounds", "histogram distribution",
    }

    # Known bivariate sections
    BIVARIATE_SUBSECTIONS = {
        "correlation", "pearson correlation", "spearman correlation",
        "kendall correlation", "mutual information", "group statistics",
        "welch t-test", "mann-whitney", "welch anova", "anova", "kruskal-wallis",
        "posthoc", "effect size", "contingency count", "overall percentage",
        "row percentage", "column percentage", "chi-square", "chi square",
        "cramér's v", "cramer's v", "fisher exact", "time trend", "spearman time",
        "temporal aggregation", "rolling statistics",
    }

    # Known multivariate sections
    MULTIVARIATE_METHODS = {
        "vif", "variance inflation factor", "pca", "principal component analysis",
        "decision tree", "decision trees", "three-way interactions",
        "three way interactions", "interaction analysis", "random forest",
        "feature importance", "permutation importance", "oob importance",
        "outliers", "clustering", "cluster analysis",
    }

    def __init__(self) -> None:
        """Initialize the EDA document parser."""
        self.logger = get_log("EDADocumentParser")
        self.logger.info("EDA document parser initialized.")

    # Public API
    @track_performance
    def parse_document(self, document: EDADocument) -> list[EDASection]:
        """
        Parse one EDA document.

        Parameters
        ----------
        document:
            Loaded EDA document.

        Returns
        -------
        list[EDASection]
            Logical EDA sections.
        """
        try:
            if not isinstance(document, EDADocument):
                raise TypeError("document must be an EDADocument.")

            self.logger.info(
                "Parsing document_id=%s, analysis_type=%s",
                document.document_id,
                document.analysis_type,
            )

            analysis_type = document.analysis_type.lower().strip()
            if analysis_type == "univariate":
                sections = self._parse_univariate(document)
            elif analysis_type == "bivariate":
                sections = self._parse_bivariate(document)
            elif analysis_type == "multivariate":
                sections = self._parse_multivariate(document)
            else:
                sections = self._parse_generic(document)

            sections = self._finalize_sections(document, sections)
            if not sections:
                sections = [self._create_fallback_section(document)]

            self.logger.info(
                "Parsed document_id=%s into %d logical sections.",
                document.document_id,
                len(sections),
            )
            return sections
        except Exception as exc:
            self.logger.error(
                "Failed to parse document_id=%s: %s",
                getattr(document, "document_id", "unknown"),
                exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def parse_documents(self, documents: list[EDADocument]) -> list[EDASection]:
        """
        Parse all EDA documents.

        Parameters
        ----------
        documents:
            Loaded EDA documents.

        Returns
        -------
        list[EDASection]
            All logical EDA sections.
        """
        try:
            if not documents:
                self.logger.warning("No EDA documents supplied.")
                return []

            sections: list[EDASection] = []
            for document in documents:
                sections.extend(self.parse_document(document))

            self.logger.info(
                "Parsed %d documents into %d total sections.",
                len(documents),
                len(sections),
            )
            return sections
        except Exception as exc:
            self.logger.error("Failed to parse EDA documents: %s", exc)
            raise CustomException(exc, self.logger) from exc

    # UNIVARIATE
    @track_performance
    def _parse_univariate(self, document: EDADocument) -> list[_SectionBuffer]:
        """
        Parse univariate EDA.

        Retrieval boundary:

            Feature + all associated statistics.

        The individual statistical subsections are retained
        inside the feature section instead of becoming separate
        retrieval documents.
        """
        sections: list[_SectionBuffer] = []
        current_feature: str | None = None
        current_lines: list[str] = []

        for line in document.content.splitlines():
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            feature_match = self.FEATURE_PATTERN.match(stripped)
            if feature_match:
                if current_feature is not None:
                    sections.append(
                        _SectionBuffer(
                            title=current_feature,
                            lines=current_lines,
                            metadata={"feature": current_feature, "section_type": "feature"},
                        )
                    )
                current_feature = feature_match.group(1).strip()
                current_lines = [stripped]
                continue

            current_lines.append(stripped)

        if current_feature is not None:
            sections.append(
                _SectionBuffer(
                    title=current_feature,
                    lines=current_lines,
                    metadata={"feature": current_feature, "section_type": "feature"},
                )
            )
        return sections

    # BIVARIATE
    @track_performance
    def _parse_bivariate(self, document: EDADocument) -> list[_SectionBuffer]:
        """
        Parse bivariate EDA.

        Retrieval boundary:

            Feature pair + all associated statistics.

        The parser preserves the subtype:

            num_num
            num_cat
            cat_cat
            num_datetime
            cat_datetime
        """
        sections: list[_SectionBuffer] = []
        current_subtype: str | None = None
        current_pair: str | None = None
        current_lines: list[str] = []

        def _flush() -> None:
            nonlocal current_pair, current_lines
            if current_pair is not None:
                sections.append(
                    self._create_bivariate_buffer(current_subtype, current_pair, current_lines)
                )

        for line in document.content.splitlines():
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            subtype_match = self.SUBTYPE_PATTERN.match(stripped)
            if subtype_match:
                _flush()
                current_subtype = subtype_match.group(1).lower()
                current_pair = None
                current_lines = [stripped]
                continue

            pair_match = self.PAIR_PATTERN.match(stripped)
            if pair_match:
                _flush()
                current_pair = pair_match.group(1).strip()
                current_lines = [stripped]
                continue

            # Some summary generators may use a plain
            # "Feature 1 vs Feature 2" heading.
            if self._looks_like_feature_pair(stripped):
                _flush()
                current_pair = stripped
                current_lines = [stripped]
                continue

            current_lines.append(stripped)

        _flush()
        return sections

    @track_performance
    def _create_bivariate_buffer(
        self,
        subtype: str | None,
        pair: str,
        lines: list[str],
    ) -> _SectionBuffer:
        """
        Create a bivariate section buffer.
        """
        metadata: dict[str, Any] = {
            "section_type": "feature_pair",
            "feature_pair": pair,
        }
        if subtype:
            metadata["subtype"] = subtype

        feature_1, feature_2 = self._split_feature_pair(pair)
        if feature_1:
            metadata["feature_1"] = feature_1
        if feature_2:
            metadata["feature_2"] = feature_2

        return _SectionBuffer(title=pair, lines=lines, metadata=metadata)

    # MULTIVARIATE
    @track_performance
    def _parse_multivariate(self, document: EDADocument) -> list[_SectionBuffer]:
        """
        Parse multivariate EDA.

        Retrieval boundary:

            One analytical method + its evidence.

        Examples:

            VIF
            PCA
            Decision Tree
            Three-Way Interactions
            Random Forest
            Clustering
        """
        sections: list[_SectionBuffer] = []
        current_method: str | None = None
        current_lines: list[str] = []

        for line in document.content.splitlines():
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            method = self._detect_multivariate_method(stripped)
            if method is not None:
                if current_method is not None:
                    sections.append(
                        _SectionBuffer(
                            title=current_method,
                            lines=current_lines,
                            metadata={"section_type": "method", "method": current_method},
                        )
                    )
                current_method = method
                current_lines = [stripped]
                continue

            current_lines.append(stripped)

        if current_method is not None:
            sections.append(
                _SectionBuffer(
                    title=current_method,
                    lines=current_lines,
                    metadata={"section_type": "method", "method": current_method},
                )
            )
        return sections

    # GENERIC FALLBACK
    @track_performance
    def _parse_generic(self, document: EDADocument) -> list[_SectionBuffer]:
        """
        Parse an unknown analysis type conservatively.

        Unknown structures are not aggressively split because
        losing statistical context is worse than retaining a
        larger section.
        """
        return [
            _SectionBuffer(
                title=f"{document.analysis_type.title()} EDA Summary",
                lines=document.content.splitlines(),
                metadata={"section_type": "document"},
            )
        ]

    # Detection helpers
    @track_performance
    def _detect_multivariate_method(self, line: str) -> str | None:
        """
        Detect a multivariate analytical method.
        """
        normalized = self._normalize_heading(line)
        for method in self.MULTIVARIATE_METHODS:
            if normalized == method:
                return line.strip()
        return None

    @track_performance
    def _looks_like_feature_pair(self, line: str) -> bool:
        """
        Detect common feature-pair heading formats.

        Examples
        --------
        tenure vs MonthlyCharges
        tenure ~ MonthlyCharges
        tenure × MonthlyCharges
        tenure and MonthlyCharges
        """
        normalized = line.lower()
        return any(sep in normalized for sep in (" vs ", " versus ", " ~ ", " × ", " and "))

    @staticmethod
    @track_performance
    def _split_feature_pair(pair: str) -> tuple[str | None, str | None]:
        """
        Split a feature pair into two feature names.
        """
        for sep in (" vs ", " versus ", " ~ ", " × ", " x ", " and "):
            parts = re.split(re.escape(sep), pair, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                first, second = parts[0].strip(), parts[1].strip()
                if first and second:
                    return first, second
        return None, None

    @staticmethod
    @track_performance
    def _normalize_heading(value: str) -> str:
        """
        Normalize a heading for comparison.
        """
        value = re.sub(r"^#{1,6}\s*", "", value.strip())
        value = value.strip(" :-=_")
        return re.sub(r"\s+", " ", value).lower()

    # Finalization
    @track_performance
    def _finalize_sections(
        self,
        document: EDADocument,
        sections: list[_SectionBuffer],
    ) -> list[EDASection]:
        """
        Convert internal buffers into EDASection objects.
        """
        result: list[EDASection] = []
        for index, buffer in enumerate(sections):
            content = self._clean_content(buffer.lines)
            if not content:
                continue

            metadata = dict(buffer.metadata)
            metadata.update({
                "source": document.source_path,
                "file_name": document.metadata.get("file_name"),
                "analysis_type": document.analysis_type,
            })

            result.append(
                EDASection(
                    section_id=f"{document.document_id}-SEC-{index:04d}",
                    document_id=document.document_id,
                    dataset_id=document.dataset_id,
                    analysis_type=document.analysis_type,
                    title=buffer.title,
                    content=content,
                    section_index=index,
                    metadata=metadata,
                )
            )
        return result

    @staticmethod
    @track_performance
    def _clean_content(lines: list[str]) -> str:
        """
        Clean whitespace while preserving structure.
        """
        result: list[str] = []
        previous_blank = False
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                if not previous_blank:
                    result.append("")
                previous_blank = True
                continue
            result.append(line.strip())
            previous_blank = False
        return "\n".join(result).strip()

    @staticmethod
    @track_performance
    def _create_fallback_section(document: EDADocument) -> EDASection:
        """
        Preserve the complete document if parsing
        cannot identify any logical boundary.
        """
        return EDASection(
            section_id=f"{document.document_id}-SEC-0000",
            document_id=document.document_id,
            dataset_id=document.dataset_id,
            analysis_type=document.analysis_type,
            title=f"{document.analysis_type.title()} EDA Summary",
            content=document.content.strip(),
            section_index=0,
            metadata={
                "source": document.source_path,
                "file_name": document.metadata.get("file_name"),
                "section_type": "fallback",
                "fallback_section": True,
            },
        )
"""EDA summary document loader."""

from __future__ import annotations

from pathlib import Path

from src.EDA_RAG.config import (
    ANALYSIS_TYPE_BIVARIATE,
    ANALYSIS_TYPE_MULTIVARIATE,
    ANALYSIS_TYPE_UNIVARIATE,
    BIVARIATE_SUMMARY_PATH,
    DEFAULT_DATASET_ID,
    MULTIVARIATE_SUMMARY_PATH,
    RAG_LOG_DIR,
    UNIVARIATE_SUMMARY_PATH,
)
from src.EDA_RAG.schemas.document_schema import EDADocument
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class EDADocumentLoader:
    """Load pre-generated EDA summary documents."""

    def __init__(self, dataset_id: str = DEFAULT_DATASET_ID) -> None:
        self.dataset_id = dataset_id.strip()
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty.")

        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("EDADocumentLoader", log_dir=RAG_LOG_DIR)

    @staticmethod
    def _build_document_id(
        analysis_type: str,
        dataset_id: str,
    ) -> str:
        suffix_map = {
            ANALYSIS_TYPE_UNIVARIATE: "UNI",
            ANALYSIS_TYPE_BIVARIATE: "BI",
            ANALYSIS_TYPE_MULTIVARIATE: "MV",
        }

        suffix = suffix_map.get(analysis_type)
        if suffix is None:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        return f"{dataset_id}-{suffix}"

    @staticmethod
    def _validate_source_file(source_path: Path) -> None:
        if not source_path.exists():
            raise FileNotFoundError(
                f"EDA summary file does not exist: {source_path}"
            )
        if not source_path.is_file():
            raise ValueError(
                f"EDA summary path is not a file: {source_path}"
            )

    def _read_file(self, source_path: Path) -> str:
        self._validate_source_file(source_path)

        content = source_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(
                f"EDA summary file is empty: {source_path}"
            )

        return content

    @track_performance
    def load_document(
        self,
        source_path: Path,
        analysis_type: str,
    ) -> EDADocument:
        try:
            source_path = Path(source_path)
            self.logger.info("Loading EDA summary: %s", source_path)

            content = self._read_file(source_path)

            document = EDADocument(
                document_id=self._build_document_id(
                    analysis_type,
                    self.dataset_id,
                ),
                dataset_id=self.dataset_id,
                analysis_type=analysis_type,
                source_path=str(source_path),
                content=content,
                metadata={
                    "file_name": source_path.name,
                    "file_extension": source_path.suffix,
                },
            )

            self.logger.info(
                "Loaded document: %s (%d characters)",
                document.document_id,
                len(content),
            )
            return document

        except Exception as exc:
            self.logger.error(
                "Failed to load EDA document from %s: %s",
                source_path,
                exc,
            )
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def load_documents(self) -> list[EDADocument]:
        try:
            configs = (
                (UNIVARIATE_SUMMARY_PATH, ANALYSIS_TYPE_UNIVARIATE),
                (BIVARIATE_SUMMARY_PATH, ANALYSIS_TYPE_BIVARIATE),
                (MULTIVARIATE_SUMMARY_PATH, ANALYSIS_TYPE_MULTIVARIATE),
            )

            documents = [
                self.load_document(path, analysis_type)
                for path, analysis_type in configs
            ]

            self.logger.info(
                "EDA summary loading completed: %d documents",
                len(documents),
            )
            return documents

        except Exception as exc:
            self.logger.error(
                "Failed to load EDA summary documents: %s",
                exc,
            )
            raise CustomException(exc, self.logger) from exc

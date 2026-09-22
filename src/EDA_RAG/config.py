"""Configuration for the EDA RAG subsystem."""

from __future__ import annotations

from pathlib import Path

from src.config import BASE_DIR


# ============================================================
# RAG directories
# ============================================================

RAG_BASE_DIR: Path = BASE_DIR / "src/EDA_RAG"
RAG_LOG_DIR: Path = RAG_BASE_DIR / "logs"
RAG_VECTOR_DB_DIR: Path = RAG_BASE_DIR / "vector_db"


# ============================================================
# EDA summaries
# ============================================================

EDA_SUMMARIES_DIR: Path = BASE_DIR / "src/Data/EDA_Reports"

UNIVARIATE_SUMMARY_PATH: Path = EDA_SUMMARIES_DIR / "univariate.txt"
BIVARIATE_SUMMARY_PATH: Path = EDA_SUMMARIES_DIR / "bivariate.txt"
MULTIVARIATE_SUMMARY_PATH: Path = EDA_SUMMARIES_DIR / "multivariate.txt"


# ============================================================
# Analysis types
# ============================================================

ANALYSIS_TYPE_UNIVARIATE: str = "univariate"
ANALYSIS_TYPE_BIVARIATE: str = "bivariate"
ANALYSIS_TYPE_MULTIVARIATE: str = "multivariate"

VALID_ANALYSIS_TYPES: tuple[str, ...] = (
    ANALYSIS_TYPE_UNIVARIATE,
    ANALYSIS_TYPE_BIVARIATE,
    ANALYSIS_TYPE_MULTIVARIATE,
)


# ============================================================
# Dataset
# ============================================================

DEFAULT_DATASET_ID: str = "DS001"


# ============================================================
# Chunking
# ============================================================

DEFAULT_CHUNK_SIZE: int = 1200
DEFAULT_CHUNK_OVERLAP: int = 150
MIN_CHUNK_SIZE: int = 100


# ============================================================
# Retrieval
# ============================================================

DEFAULT_RETRIEVAL_TOP_K: int = 10
DEFAULT_RERANK_TOP_K: int = 5


# ============================================================
# Embeddings
# ============================================================

EMBEDDING_PROVIDER: str = "hf-inference"
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

EMBEDDING_BATCH_SIZE: int = 32
EMBEDDING_NORMALIZE: bool = True


# ============================================================
# Vector store
# ============================================================

FAISS_INDEX_NAME: str = "eda_rag.index"
METADATA_FILE_NAME: str = "eda_rag_metadata.json"


# ============================================================
# Logging
# ============================================================

RAG_LOGGER_NAME: str = "EDA_RAG"

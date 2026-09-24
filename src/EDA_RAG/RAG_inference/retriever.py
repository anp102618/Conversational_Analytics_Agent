"""
EDA RAG retrieval service.

Responsibilities:
    1. Initialize the query embedding service.
    2. Load the persisted FAISS vector store.
    3. Convert a user query into an embedding.
    4. Retrieve the most relevant EDA chunks.
    5. Return structured RetrievalResponse.

The service performs vector retrieval only.

It does not:
    - rerank results,
    - perform query expansion,
    - perform BM25/hybrid retrieval,
    - call an LLM,
    - generate the final answer.
"""
from __future__ import annotations

from typing import Optional

from src.EDA_RAG.config import (
    DEFAULT_DATASET_ID,
    DEFAULT_RETRIEVAL_TOP_K,
    RAG_LOG_DIR,
)
from src.EDA_RAG.RAG_build.embedder import EmbeddingService
from src.EDA_RAG.schemas.retrieval_schema import RetrievalResponse
from src.EDA_RAG.RAG_build.vector_store import FAISSVectorStore
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class Retrieval_Service:
    """
    Retrieve relevant EDA evidence from the FAISS vector database.

    The Retriever owns the query-time retrieval dependencies:
        - EmbeddingService
        - FAISSVectorStore

    This keeps the baseline retrieval implementation simple while
    remaining independent from the later reranking and generation
    components.
    """

    def __init__(self, dataset_id: str = DEFAULT_DATASET_ID, logger=None) -> None:
        """
        Initialize the Retriever.

        Args:
            dataset_id: Default dataset used for retrieval.

        Raises:
            CustomException: If the embedding service or vector store
                cannot be initialized.
            ValueError: If dataset_id is empty.
        """
        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_log("RAGInference", log_dir=RAG_LOG_DIR)

        try:
            if not isinstance(dataset_id, str):
                raise TypeError("dataset_id must be a string.")
            dataset_id = dataset_id.strip()
            if not dataset_id:
                raise ValueError("dataset_id cannot be empty.")

            self.dataset_id = dataset_id
            self.logger.info(
                "Initializing Retriever | dataset_id=%s", self.dataset_id
            )

            # Query embedding service.
            self.embedding_service = EmbeddingService()

            # FAISS vector store.
            # Use the same constructor configuration used by your
            # vector DB build process.
            self.vector_store = FAISSVectorStore()

            # Load persisted FAISS index and metadata.
            self.vector_store.load()
            self.logger.info(
                "FAISS vector store loaded | vectors=%d",
                self.vector_store.count,
            )
            self.logger.info("Retriever initialized successfully.")
        except Exception as exc:
            self.logger.error("Failed to initialize Retriever: %s", exc)
            raise CustomException(exc, self.logger) from exc

    # Validation helpers
    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate and normalize the query.

        Args:
            query: User's natural-language query.

        Returns:
            Stripped query.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")
        return query

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        """
        Validate the number of retrieval candidates.

        Args:
            top_k: Number of FAISS results to retrieve.

        Returns:
            Validated top_k.

        Raises:
            TypeError: If top_k is not an integer.
            ValueError: If top_k is not positive.
        """
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        return top_k

    @staticmethod
    def _validate_dataset_id(dataset_id: Optional[str]) -> Optional[str]:
        """
        Validate an optional dataset ID.

        Args:
            dataset_id: Dataset filter.

        Returns:
            Stripped dataset ID or None.

        Raises:
            TypeError: If dataset_id is not a string.
            ValueError: If dataset_id is empty.
        """
        if dataset_id is None:
            return None
        if not isinstance(dataset_id, str):
            raise TypeError("dataset_id must be a string.")
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty.")
        return dataset_id

    @staticmethod
    def _validate_analysis_type(analysis_type: Optional[str]) -> Optional[str]:
        """
        Validate an optional EDA analysis type.

        Args:
            analysis_type: EDA analysis type.

        Returns:
            Normalized analysis type or None.

        Raises:
            TypeError: If analysis_type is not a string.
            ValueError: If analysis_type is unsupported.
        """
        if analysis_type is None:
            return None
        if not isinstance(analysis_type, str):
            raise TypeError("analysis_type must be a string.")
        analysis_type = analysis_type.strip().lower()
        valid_types = {"univariate", "bivariate", "multivariate"}
        if analysis_type not in valid_types:
            raise ValueError(
                f"Unsupported analysis_type: {analysis_type}. "
                f"Expected one of: {sorted(valid_types)}"
            )
        return analysis_type

    # Retrieval
    @track_performance
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        dataset_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant EDA chunks for a user query.

        Retrieval flow:

            Query
              ↓
            Query Embedding
              ↓
            FAISS Similarity Search
              ↓
            RetrievalResponse

        Args:
            query: Natural-language user query.
            top_k: Number of FAISS candidates to retrieve.
            dataset_id: Optional dataset filter. If omitted, the
                Retriever's default dataset is used.
            analysis_type: Optional EDA type filter:
                univariate, bivariate, or multivariate.

        Returns:
            RetrievalResponse containing the query, dataset ID,
            and ranked retrieval results.

        Raises:
            CustomException: If embedding or retrieval fails.
            ValueError: If an input is invalid.
        """
        try:
            # Validate inputs.
            query = self._validate_query(query)
            top_k = self._validate_top_k(top_k)
            dataset_id = self._validate_dataset_id(dataset_id)
            analysis_type = self._validate_analysis_type(analysis_type)
            effective_dataset_id = (
                dataset_id if dataset_id is not None else self.dataset_id
            )

            # Log request.
            self.logger.info(
                "Starting retrieval | query=%r | top_k=%d | dataset_id=%s | analysis_type=%s",
                query,
                top_k,
                effective_dataset_id,
                analysis_type,
            )

            # Step 1: Generate query embedding.
            query_embedding = self.embedding_service.embed_text(query)
            self.logger.info(
                "Query embedding generated | dimension=%d",
                query_embedding.shape[0],
            )

            # Step 2: Search FAISS.
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                dataset_id=effective_dataset_id,
                analysis_type=analysis_type,
            )
            self.logger.info(
                "FAISS retrieval completed | results=%d", len(results)
            )

            # Step 3: Handle empty retrieval.
            if not results:
                self.logger.warning(
                    "No relevant EDA chunks found | query=%r | dataset_id=%s | analysis_type=%s",
                    query,
                    effective_dataset_id,
                    analysis_type,
                )

            # Step 4: Construct structured response.
            return RetrievalResponse(
                query=query,
                dataset_id=effective_dataset_id,
                results=results,
            )
        except Exception as exc:
            self.logger.error("Retrieval failed for query=%r: %s", query, exc)
            raise CustomException(exc, self.logger) from exc



rag_retriever = Retrieval_Service()
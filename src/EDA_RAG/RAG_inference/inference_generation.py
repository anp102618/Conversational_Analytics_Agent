
"""
EDA RAG inference pipeline.

Serially orchestrates:
1. Query expansion
2. Vector retrieval
3. Cross-encoder reranking
4. Grounded answer generation

Each stage is implemented independently and imported here.
"""
from __future__ import annotations

from src.EDA_RAG.RAG_inference.generator import rag_generator
from src.EDA_RAG.RAG_inference.query_expander import rag_query_expander
from src.EDA_RAG.RAG_inference.reranker import rag_reranker
from src.EDA_RAG.RAG_inference.retriever import rag_retriever
from src.EDA_RAG.config import (
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    RAG_LOG_DIR,
)
from src.EDA_RAG.schemas.retrieval_schema import RetrievalResponse
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import current_logger, get_log, track_performance


class RAGInference:
    """
    End-to-end EDA RAG inference pipeline.

    Pipeline:
        User Query
            ↓
        Query Expansion
            ↓
        Vector Retrieval
            ↓
        Cross-Encoder Reranking
            ↓
        Grounded Answer Generation
            ↓
        Final Answer

    Query expansion is used only for retrieval.
    The original user query is used for:
        - reranking
        - final answer generation

    A single logger is shared across all RAG components.
    """

    def __init__(
        self,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> None:
        """
        Initialize the RAG inference pipeline.

        Args:
            retrieval_top_k: Number of candidates retrieved from FAISS.
            rerank_top_k: Number of candidates retained after reranking.

        Raises:
            ValueError: If either top-k value is invalid.
        """
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1.")
        if rerank_top_k < 1:
            raise ValueError("rerank_top_k must be at least 1.")
        if rerank_top_k > retrieval_top_k:
            raise ValueError(
                "rerank_top_k cannot be greater than retrieval_top_k."
            )

        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("RAGInference", log_dir=RAG_LOG_DIR)
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k

        # Shared RAG components.
        self.query_expander = rag_query_expander
        self.retriever = rag_retriever
        self.reranker = rag_reranker
        self.generator = rag_generator
        self._configure_component_loggers()

        self.logger.info(
            "RAG inference pipeline initialized. retrieval_top_k=%d, rerank_top_k=%d",
            self.retrieval_top_k,
            self.rerank_top_k,
        )

    def _configure_component_loggers(self) -> None:
        """
        Assign the shared RAG logger to all inference components.

        This ensures query expansion, retrieval, reranking, and
        generation write to the same RAG log file.
        """
        self.query_expander.logger = self.logger
        self.retriever.logger = self.logger
        self.reranker.logger = self.logger
        self.generator.logger = self.logger

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate and clean the user query.

        Args:
            query: User's analytical question.

        Returns:
            Cleaned query.

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
    def _validate_analysis_type(analysis_type: str | None) -> str | None:
        """
        Validate the optional EDA analysis type.

        Args:
            analysis_type: Expected to be univariate, bivariate,
                multivariate, or None.

        Returns:
            Normalized analysis type or None.

        Raises:
            TypeError: If analysis_type is not a string or None.
            ValueError: If the analysis type is invalid.
        """
        if analysis_type is None:
            return None
        if not isinstance(analysis_type, str):
            raise TypeError("analysis_type must be a string or None.")
        analysis_type = analysis_type.strip().lower()
        if not analysis_type:
            return None
        valid_types = {"univariate", "bivariate", "multivariate"}
        if analysis_type not in valid_types:
            raise ValueError(
                f"Invalid analysis_type={analysis_type!r}. "
                f"Expected one of {sorted(valid_types)}."
            )
        return analysis_type

    def run(
        self,
        query: str,
        analysis_type: str | None = None,
        dataset_id: str | None = None,
    ) -> str:
        """
        Execute the complete EDA RAG inference pipeline.

        This public method establishes the RAG logger as the active
        pipeline logger before calling the performance-tracked method.

        Args:
            query: Original user question.
            analysis_type: Optional EDA type filter.
            dataset_id: Optional dataset identifier.

        Returns:
            Final grounded answer.

        Raises:
            CustomException: If any pipeline stage fails.
        """
        logger_token = current_logger.set(self.logger)
        try:
            return self._run(
                query=query,
                analysis_type=analysis_type,
                dataset_id=dataset_id,
            )
        finally:
            current_logger.reset(logger_token)

    @track_performance
    def _run(
        self,
        query: str,
        analysis_type: str | None = None,
        dataset_id: str | None = None,
    ) -> str:
        """
        Execute the performance-tracked RAG inference pipeline.

        The active logger context is established by ``run`` before
        this method is invoked.
        """
        try:
            query = self._validate_query(query)
            analysis_type = self._validate_analysis_type(analysis_type)

            self.logger.info("==================================================")
            self.logger.info("Starting RAG inference.")
            self.logger.info("Query: %r", query)
            self.logger.info("Dataset ID: %s", dataset_id or "default")
            self.logger.info("Analysis type: %s", analysis_type or "all")

            # 1. QUERY EXPANSION
            self.logger.info("[1/4] Starting query expansion.")
            expanded_query = self.query_expander.expand(query=query)
            self.logger.info("[1/4] Query expansion completed.")
            self.logger.debug("Original query: %s", query)
            self.logger.debug("Expanded query: %s", expanded_query)

            # 2. VECTOR RETRIEVAL
            self.logger.info("[2/4] Starting vector retrieval.")
            retrieval_response: RetrievalResponse = self.retriever.retrieve(
                query=expanded_query,
                top_k=self.retrieval_top_k,
                dataset_id=dataset_id,
                analysis_type=analysis_type,
            )
            self.logger.info(
                "[2/4] Retrieval completed. Retrieved=%d results.",
                len(retrieval_response.results),
            )
            if not retrieval_response.results:
                self.logger.warning("No retrieval results found.")
                return (
                    "I could not find sufficient EDA evidence to answer the question."
                )

            # 3. CROSS-ENCODER RERANKING
            self.logger.info("[3/4] Starting cross-encoder reranking.")
            reranked_results = self.reranker.rerank(
                query=query,
                results=retrieval_response.results,
                top_k=self.rerank_top_k,
            )
            self.logger.info(
                "[3/4] Reranking completed. Retained=%d results.",
                len(reranked_results),
            )
            if not reranked_results:
                self.logger.warning("No results remained after reranking.")
                return (
                    "I could not find sufficiently relevant EDA evidence "
                    "to answer the question."
                )

            # 4. GROUNDED GENERATION
            self.logger.info("[4/4] Starting grounded answer generation.")
            answer = self.generator.generate(query=query, results=reranked_results)
            self.logger.info("[4/4] RAG generation completed successfully.")
            self.logger.info("RAG inference completed successfully.")
            self.logger.info("==================================================")
            return answer
        except Exception as exc:
            self.logger.error("RAG inference failed: %s", exc)
            raise CustomException(exc, self.logger) from exc


rag_inference = RAGInference()
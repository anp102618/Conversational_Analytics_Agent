"""
Cross-encoder reranking service for EDA RAG.

The reranker receives the initial FAISS retrieval results
and reorders them using a cross-encoder relevance model.

Flow:
    User Query
        ↓
    FAISS Top-K Results
        ↓
    Cross-Encoder
        ↓
    Reranking Scores
        ↓
    Sorted Top-N Results

The reranker does not:
    - perform vector retrieval,
    - generate answers,
    - call an LLM,
    - perform query expansion,
    - perform hybrid retrieval.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.EDA_RAG.config import DEFAULT_RERANK_TOP_K, RAG_LOG_DIR
from src.EDA_RAG.schemas.retrieval_schema import RetrievalResult
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank retrieved EDA chunks using a cross-encoder.

    The original vector similarity score is preserved in the
    RetrievalResult. The cross-encoder score is stored separately
    in the result metadata.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        logger=None
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Hugging Face cross-encoder model.
            max_length: Maximum sequence length.

        Raises:
            ValueError: If model_name or max_length is invalid.
            CustomException: If the model cannot be loaded.
        """
        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_log("RAGInference", log_dir=RAG_LOG_DIR)

        try:
            if not isinstance(model_name, str):
                raise TypeError("model_name must be a string.")
            model_name = model_name.strip()
            if not model_name:
                raise ValueError("model_name cannot be empty.")
            if not isinstance(max_length, int):
                raise TypeError("max_length must be an integer.")
            if max_length <= 0:
                raise ValueError("max_length must be greater than zero.")

            self.model_name = model_name
            self.max_length = max_length
            self.logger.info("Loading cross-encoder model: %s", self.model_name)
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            self.logger.info(
                "Cross-encoder loaded successfully | model=%s", self.model_name
            )
        except Exception as exc:
            self.logger.error("Failed to initialize CrossEncoderReranker: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate the query.

        Args:
            query: User query.

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
    def _validate_results(results: list[RetrievalResult]) -> None:
        """
        Validate retrieved results.

        Args:
            results: Initial FAISS results.

        Raises:
            TypeError: If results is not a list.
            ValueError: If an item is not RetrievalResult.
        """
        if not isinstance(results, list):
            raise TypeError("results must be a list.")
        for result in results:
            if not isinstance(result, RetrievalResult):
                raise ValueError("All results must be RetrievalResult instances.")

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        """
        Validate reranking top-k.

        Args:
            top_k: Number of results to return.

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
    def _update_metadata(
        result: RetrievalResult,
        rerank_score: float,
        initial_rank: int,
    ) -> None:
        """
        Store reranking information without overwriting FAISS score.

        Args:
            result: Retrieval result.
            rerank_score: Cross-encoder relevance score.
            initial_rank: Rank assigned by FAISS.
        """
        if result.metadata is None:
            result.metadata = {}
        result.metadata["initial_rank"] = initial_rank
        result.metadata["rerank_score"] = float(rerank_score)

    @track_performance
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RetrievalResult]:
        """
        Rerank FAISS results using the cross-encoder.

        Args:
            query: Original user query.
            results: Results returned by the Retriever.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked RetrievalResult objects.

        Raises:
            CustomException: If reranking fails.
        """
        try:
            query = self._validate_query(query)
            self._validate_results(results)
            top_k = self._validate_top_k(top_k)

            if not results:
                self.logger.warning("No retrieval results available for reranking.")
                return []

            effective_top_k = min(top_k, len(results))
            self.logger.info(
                "Starting reranking | candidates=%d | top_k=%d",
                len(results),
                effective_top_k,
            )

            # Preserve the original FAISS ranking.
            for rank, result in enumerate(results, start=1):
                if result.rank is None:
                    result.rank = rank

            # Build query-document pairs.
            pairs = [(query, result.content) for result in results]

            # Cross-encoder inference.
            scores = self.model.predict(pairs, show_progress_bar=False)
            # Convert numpy/scalar output into Python floats.
            rerank_scores = [float(score) for score in scores]

            # Attach reranking scores while preserving FAISS scores.
            scored_results: list[tuple[RetrievalResult, float]] = []
            for index, result in enumerate(results):
                rerank_score = rerank_scores[index]
                initial_rank = result.rank if result.rank is not None else index + 1
                self._update_metadata(
                    result=result,
                    rerank_score=rerank_score,
                    initial_rank=initial_rank,
                )
                scored_results.append((result, rerank_score))

            # Sort by cross-encoder score.
            scored_results.sort(key=lambda item: item[1], reverse=True)
            reranked_results = [
                result for result, _ in scored_results[:effective_top_k]
            ]

            # Assign the new final ranking.
            for rank, result in enumerate(reranked_results, start=1):
                result.rank = rank

            self.logger.info(
                "Reranking completed | returned=%d", len(reranked_results)
            )

            # Log top result for debugging/observability.
            if reranked_results:
                top_result = reranked_results[0]
                self.logger.info(
                    "Top reranked result | chunk_id=%s | vector_score=%.4f | rerank_score=%.4f",
                    top_result.chunk_id,
                    float(top_result.score),
                    float(top_result.metadata["rerank_score"]),
                )

            return reranked_results
        except Exception as exc:
            self.logger.error("Reranking failed for query=%r: %s", query, exc)
            raise CustomException(exc, self.logger) from exc


rag_reranker = CrossEncoderReranker()
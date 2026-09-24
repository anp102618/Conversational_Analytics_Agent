
"""
EDA RAG query expansion.

Expands a user's analytical question into a retrieval-friendly query
while preserving the original intent.

NVIDIA NIM is used as the primary LLM and Gemini Flash-Lite is used
as the fallback.
"""
from __future__ import annotations

from src.config import config
from src.EDA_RAG.config import RAG_LOG_DIR
from src.Utils.llm_variants import (
    GeminiFlashLiteStrategy,
    NvidiaNimStrategy,
    ResilientLLMContext,
)
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class QueryExpansion:
    """
    Expand analytical queries for improved EDA retrieval.

    The class reuses the existing LLM strategy/failover implementation:
        NVIDIA NIM
            ↓ failure after retries
        Gemini Flash-Lite

    The expansion is intentionally conservative. It should add useful
    analytical terminology without changing the user's intent.
    """

    DEFAULT_MAX_RETRIES = 3

    SYSTEM_PROMPT = """
You are a query expansion component for an Exploratory Data Analysis
(EDA) retrieval system.

Your task is to transform the user's analytical question into a single
retrieval-friendly query.

Rules:
1. Preserve the exact intent of the original question.
2. Do not answer the question.
3. Do not invent statistics, values, correlations, or findings.
4. Do not introduce variables that are unrelated to the question.
5. Preserve all explicitly mentioned variable names.
6. Add relevant analytical terminology when useful.
7. For relationship questions, include appropriate terms such as
   correlation, association, relationship, trend, or dependence when
   relevant.
8. For distribution questions, include relevant terms such as
   mean, median, variance, standard deviation, skewness, kurtosis,
   quantiles, or outliers only when appropriate.
9. For missing-value questions, include terms related to missing,
   null, percentage, and completeness when appropriate.
10. For categorical questions, include relevant frequency,
    proportion, percentage, or category-distribution terminology.
11. Do not convert correlation into causation.
12. Return ONLY the expanded query.
13. Do not add explanations, labels, bullet points, or quotation marks.
""".strip()

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES, logger=None) -> None:
        """
        Initialize the query expansion service.

        Args:
            max_retries: Number of retries for each LLM before the
                fallback model is attempted.

        Raises:
            ValueError: If max_retries is less than 1.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logger or get_log("RAGInference", log_dir=RAG_LOG_DIR)
        self.max_retries = max_retries
        self._initialize_llm()

    @track_performance
    def _initialize_llm(self) -> None:
        """
        Initialize NVIDIA NIM as primary and Gemini as fallback.
        """
        try:
            self.nvidia_strategy = NvidiaNimStrategy(
                api_key=config.NVIDIA_API_KEY,
                model=config.NVIDIA_MODEL,
            )
            self.gemini_strategy = GeminiFlashLiteStrategy(
                api_key=config.GEMINI_API_KEY,
                model="gemini-3.1-flash-lite",
            )
            self.llm = ResilientLLMContext(
                primary_llm=self.nvidia_strategy,
                fallback_llm=self.gemini_strategy,
                max_retries=self.max_retries,
            )
            self.logger.info(
                "Query expansion LLM initialized. "
                "Primary=NVIDIA NIM, Fallback=Gemini Flash-Lite."
            )
        except Exception as exc:
            self.logger.error("Failed to initialize query expansion LLM: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @staticmethod
    @track_performance
    def _validate_query(query: str) -> str:
        """
        Validate the incoming user query.

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
    @track_performance
    def _clean_response(response: str) -> str:
        """
        Clean the generated expansion.

        Removes common formatting accidentally produced by the LLM.
        """
        if not isinstance(response, str):
            raise TypeError("LLM response must be a string.")
        response = response.strip()
        if not response:
            raise ValueError("LLM returned an empty query expansion.")

        # Remove accidental surrounding quotation marks.
        if (
            len(response) >= 2
            and response[0] == response[-1]
            and response[0] in {'"', "'"}
        ):
            response = response[1:-1].strip()

        # Remove accidental labels.
        for prefix in ("Expanded Query:", "Expanded query:", "Query:"):
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break

        if not response:
            raise ValueError("Query expansion became empty after cleaning.")
        return response

    @track_performance
    def expand(self, query: str) -> str:
        """
        Expand an analytical query.

        Args:
            query: Original user question.

        Returns:
            Retrieval-friendly expanded query.

        Raises:
            CustomException: If query expansion fails after both
                primary and fallback LLM attempts.
        """
        try:
            query = self._validate_query(query)
            self.logger.info("Expanding query: %r", query)

            expanded_query = self.llm.generate(
                prompt=query,
                system_instruction=self.SYSTEM_PROMPT,
            )
            expanded_query = self._clean_response(expanded_query)

            self.logger.info("Query expansion completed: %r", expanded_query)
            return expanded_query
        except Exception as exc:
            self.logger.error("Query expansion failed: %s", exc)
            raise CustomException(exc, self.logger) from exc


rag_query_expander = QueryExpansion()


"""
EDA RAG generation service.

Builds the generation prompt from reranked EDA evidence and generates
a grounded answer using NVIDIA NIM as the primary LLM with Gemini
Flash-Lite as the fallback.
"""
from __future__ import annotations

from typing import Generator, Sequence

from src.EDA_RAG.config import RAG_LOG_DIR
from src.EDA_RAG.schemas.retrieval_schema import RetrievalResult
from src.Utils.llm_variants import (
    GeminiFlashLiteStrategy,
    NvidiaNimStrategy,
    ResilientLLMContext,
)
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class RAGGenerator:
    """
    Generate grounded answers from reranked EDA retrieval results.

    The generator is responsible for:

    1. Converting reranked results into evidence.
    2. Building the system prompt.
    3. Building the user prompt.
    4. Calling NVIDIA NIM as the primary LLM.
    5. Falling back to Gemini Flash-Lite when NVIDIA fails.
    """

    DEFAULT_MAX_RETRIES = 3

    SYSTEM_PROMPT = """
You are an EDA analysis assistant.

Your task is to answer the user's question using ONLY the supplied
EDA evidence.

Grounding rules:
1. Use only information contained in the supplied evidence.
2. Do not invent statistics, relationships, values, or conclusions.
3. Do not use external knowledge.
4. Preserve numerical values exactly as supported by the evidence.
5. Distinguish correlation from causation.
6. If the evidence is insufficient to answer the question, explicitly
   state that the available EDA evidence is insufficient.
7. Do not claim a relationship is statistically significant unless
   the evidence explicitly supports statistical significance.
8. When describing a relationship, identify the relevant variables
   and the available statistical evidence.
9. Prefer precise numerical evidence over vague descriptions.
10. Keep the answer concise but sufficiently explanatory.
11. Do not mention the retrieval system, vector database, reranker,
    prompts, or internal implementation unless explicitly asked.
""".strip()

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES, logger=None) -> None:
        """
        Initialize the RAG generator.

        Args:
            max_retries: Number of retries attempted for each LLM
                before failing over to the fallback model.

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
        """Initialize primary and fallback LLM strategies."""
        try:
            from src.config import config

            self.nvidia_strategy = NvidiaNimStrategy(
                api_key=config.NVIDIA_API_KEY,
                model=config.NVIDIA_MODEL,
            )
            self.gemini_strategy = GeminiFlashLiteStrategy(
                api_key=config.GEMINI_API_KEY,
                model=config.GEMINI_MODEL
            )
            self.llm = ResilientLLMContext(
                primary_llm=self.nvidia_strategy,
                fallback_llm=self.gemini_strategy,
                max_retries=self.max_retries,
            )
            self.logger.info(
                "RAG LLM initialized. Primary=NVIDIA NIM, Fallback=Gemini Flash-Lite."
            )
        except Exception as exc:
            self.logger.error("Failed to initialize RAG LLM: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @staticmethod
    @track_performance
    def _validate_query(query: str) -> str:
        """Validate and normalize the user query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty.")
        return query

    @staticmethod
    @track_performance
    def _validate_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
        """Validate reranked retrieval results."""
        if results is None:
            raise ValueError("results cannot be None.")
        results = list(results)
        if not results:
            raise ValueError("No retrieval evidence was supplied for generation.")
        return results

    @staticmethod
    @track_performance
    def _build_evidence(results: Sequence[RetrievalResult]) -> str:
        """
        Convert reranked retrieval results into grounded evidence.

        The original retrieval score is not used to construct the
        textual evidence. The reranked order determines the evidence
        order.
        """
        evidence_blocks = [
            f"[Evidence {index}]\nChunk ID: {result.chunk_id}\n{result.content}"
            for index, result in enumerate(results, start=1)
        ]
        return "\n\n".join(evidence_blocks)

    @classmethod
    @track_performance
    def _build_user_prompt(
        cls,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        """
        Build the user prompt containing the question and EDA evidence.
        """
        evidence = cls._build_evidence(results)
        return f"""
User Question:
{query}

EDA Evidence:
{evidence}

Instructions:
Answer the user question using only the EDA evidence above.

If the evidence directly supports a numerical value, include that
value in the answer.

If multiple pieces of evidence are relevant, synthesize them without
introducing unsupported conclusions.

If the evidence does not sufficiently answer the question, state that
the available EDA evidence is insufficient.

Do not invent missing statistics.
""".strip()

    @track_performance
    def generate(
        self,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        """
        Generate a grounded answer from reranked EDA evidence.

        Args:
            query: Original user question.
            results: Reranked EDA retrieval results.

        Returns:
            Generated grounded answer.

        Raises:
            CustomException: If generation fails after primary and
                fallback LLM attempts.
        """
        try:
            query = self._validate_query(query)
            results = self._validate_results(results)
            user_prompt = self._build_user_prompt(query=query, results=results)

            self.logger.info(
                "Generating RAG answer using %d evidence chunks.", len(results)
            )
            answer = self.llm.generate(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
            )
            if not answer or not answer.strip():
                raise ValueError("LLM returned an empty response.")

            self.logger.info("RAG answer generated successfully.")
            return answer.strip()
        except Exception as exc:
            self.logger.error("RAG generation failed: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def generate_stream(
        self,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> Generator[str, None, None]:
        """
        Stream a grounded answer from reranked EDA evidence.

        Args:
            query: Original user question.
            results: Reranked EDA retrieval results.

        Yields:
            Generated answer chunks.

        Raises:
            CustomException: If streaming generation fails.
        """
        try:
            query = self._validate_query(query)
            results = self._validate_results(results)
            user_prompt = self._build_user_prompt(query=query, results=results)

            self.logger.info(
                "Starting streaming RAG generation using %d evidence chunks.",
                len(results),
            )
            yield from self.llm.generate_stream(
                prompt=user_prompt,
                system_instruction=self.SYSTEM_PROMPT,
            )
        except Exception as exc:
            self.logger.error("RAG streaming generation failed: %s", exc)
            raise CustomException(exc, self.logger) from exc


rag_generator = RAGGenerator()

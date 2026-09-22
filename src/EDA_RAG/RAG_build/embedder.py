"""
Embedding service for the EDA RAG subsystem.

Uses the open-source:
    sentence-transformers/all-mpnet-base-v2

through the Hugging Face Inference API.

The service:
- Generates embeddings for individual text inputs.
- Generates embeddings for EDA chunks.
- Normalizes embeddings for cosine similarity.
- Validates embedding dimensions and numerical values.
- Returns NumPy float32 vectors suitable for FAISS.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from huggingface_hub import InferenceClient

from src.EDA_RAG.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_PROVIDER,
)
from src.EDA_RAG.schemas.chunk_schema import EDAChunk
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance
from src.config import config


class EmbeddingService:
    """
    Generate embeddings using Hugging Face Inference API.

    Parameters
    ----------
    model:
        Hugging Face embedding model.
    provider:
        Hugging Face inference provider.
    batch_size:
        Number of chunks processed in one logical batch.
    normalize:
        Whether to L2-normalize embeddings.

    Notes
    -----
    The default model is:
        sentence-transformers/all-mpnet-base-v2

    The model produces 768-dimensional embeddings.
    The dimension is discovered dynamically rather than
    hardcoded so the service can support other embedding
    models in the future.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        provider: str = EMBEDDING_PROVIDER,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize: bool = EMBEDDING_NORMALIZE,
    ) -> None:
        """Initialize the embedding service."""
        if not model.strip():
            raise ValueError("Embedding model cannot be empty.")
        if not provider.strip():
            raise ValueError("Embedding provider cannot be empty.")
        if batch_size <= 0:
            raise ValueError("Embedding batch_size must be greater than zero.")

        token = config.HF_TOKEN
        if not token:
            raise EnvironmentError("HF_TOKEN environment variable is not configured.")

        self.model = model
        self.provider_name = provider
        self.batch_size = batch_size
        self.normalize = normalize
        self.client = InferenceClient(provider=self.provider_name, api_key=token)
        self.embedding_dimension: int | None = None
        self.logger = get_log("EmbeddingService")
        self.logger.info(
            "Embedding service initialized: model=%s | provider=%s | batch_size=%d | normalize=%s",
            self.model,
            self.provider_name,
            self.batch_size,
            self.normalize,
        )

    # Public Methods
    @track_performance
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Parameters
        ----------
        text:
            Text to embed.

        Returns
        -------
        numpy.ndarray
            One-dimensional float32 embedding vector.
        """
        try:
            text = self._validate_text(text)
            response = self.client.feature_extraction(text, model=self.model)
            vector = self._prepare_vector(self._convert_response(response))
            self._validate_dimension(vector)
            return vector
        except Exception as exc:
            self.logger.error("Failed to generate text embedding: %s", exc)
            raise CustomException(exc, self.logger) from exc

    @track_performance
    def embed_chunks(self, chunks: list[EDAChunk]) -> np.ndarray:
        """
        Generate embeddings for EDA chunks.

        Parameters
        ----------
        chunks:
            List of EDAChunk objects.

        Returns
        -------
        numpy.ndarray
            Two-dimensional float32 matrix with shape:
                (number_of_chunks, embedding_dimension)
        """
        try:
            if not chunks:
                self.logger.warning("No chunks supplied for embedding.")
                return np.empty((0, 0), dtype=np.float32)

            self._validate_chunks(chunks)
            self.logger.info("Generating embeddings for %d EDA chunks.", len(chunks))

            vectors: list[np.ndarray] = []
            for start in range(0, len(chunks), self.batch_size):
                end = min(start + self.batch_size, len(chunks))
                batch = chunks[start:end]
                self.logger.debug("Embedding batch: %d-%d", start, end - 1)
                vectors.extend(self._embed_batch(batch))

            matrix = np.vstack(vectors).astype(np.float32)
            self._validate_matrix(matrix, expected_rows=len(chunks))
            self.logger.info(
                "Embedding completed successfully: chunks=%d | dimension=%d",
                matrix.shape[0],
                matrix.shape[1],
            )
            return matrix
        except Exception as exc:
            self.logger.error("Failed to generate chunk embeddings: %s", exc)
            raise CustomException(exc, self.logger) from exc

    # Internal Embedding Methods
    def _embed_batch(self, chunks: list[EDAChunk]) -> list[np.ndarray]:
        """
        Generate embeddings for one logical batch.

        The current implementation performs one API request
        per chunk. The batch size controls application-level
        processing and can later be replaced with true API
        batching without changing the public interface.
        """
        vectors: list[np.ndarray] = []
        for chunk in chunks:
            response = self.client.feature_extraction(chunk.content, model=self.model)
            vector = self._prepare_vector(self._convert_response(response))
            self._validate_dimension(vector)
            vectors.append(vector)
        return vectors

    # Response Conversion
    @staticmethod
    def _convert_response(response: Any) -> np.ndarray:
        """
        Convert Hugging Face feature-extraction output
        into a one-dimensional NumPy vector.

        Supported shapes:
            [dimension]
        or:
            [[dimension]]
        """
        vector = np.asarray(response, dtype=np.float32)
        if vector.ndim == 2:
            if vector.shape[0] != 1:
                raise ValueError(
                    f"Expected a single embedding vector, received shape={vector.shape}."
                )
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(f"Embedding must be one-dimensional. Received shape={vector.shape}.")
        if vector.size == 0:
            raise ValueError("Embedding vector is empty.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains NaN or infinite values.")
        return vector.astype(np.float32)

    # Vector Preparation
    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Validate and optionally L2-normalize an embedding.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Embedding vector must be one-dimensional.")
        if vector.size == 0:
            raise ValueError("Embedding vector cannot be empty.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains invalid numerical values.")
        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("Cannot normalize a zero embedding vector.")
            vector = vector / norm
        return vector.astype(np.float32)

    # Dimension Validation
    def _validate_dimension(self, vector: np.ndarray) -> None:
        """
        Establish the embedding dimension from the first
        vector and validate subsequent vectors.
        """
        dimension = int(vector.shape[0])
        if self.embedding_dimension is None:
            self.embedding_dimension = dimension
            self.logger.info("Detected embedding dimension: %d", dimension)
            return
        if dimension != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected={self.embedding_dimension}, "
                f"received={dimension}."
            )

    # Input Validation
    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate text before sending it to the API.
        """
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("Cannot generate embedding for empty text.")
        return text

    @staticmethod
    def _validate_chunks(chunks: list[EDAChunk]) -> None:
        """
        Validate EDA chunks before embedding.
        """
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, EDAChunk):
                raise TypeError(f"Item {index} is not an EDAChunk.")
            if not chunk.content.strip():
                raise ValueError(f"Chunk {index} contains empty content.")

    @staticmethod
    def _validate_matrix(matrix: np.ndarray, expected_rows: int) -> None:
        """
        Validate the final embedding matrix.
        """
        if matrix.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional.")
        if matrix.shape[0] != expected_rows:
            raise ValueError("Embedding count does not match the number of chunks.")
        if matrix.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")
        if not np.isfinite(matrix).all():
            raise ValueError("Embedding matrix contains invalid values.")

    # Properties
    @property
    def dimension(self) -> int | None:
        """
        Return the detected embedding dimension.
        """
        return self.embedding_dimension
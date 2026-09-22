"""
Build the EDA RAG vector database.

Pipeline
--------
EDA summary files
    ↓
EDADocumentLoader
    ↓
EDADocumentParser
    ↓
EDAChunker
    ↓
EmbeddingService
    ↓
FAISSVectorStore
    ↓
Persistent vector database
"""
from __future__ import annotations

from typing import Any

from src.EDA_RAG.config import DEFAULT_DATASET_ID, RAG_LOG_DIR
from src.EDA_RAG.RAG_build.chunker import EDASemanticChunker
from src.EDA_RAG.RAG_build.embedder import EmbeddingService
from src.EDA_RAG.RAG_build.document_loader import EDADocumentLoader
from src.EDA_RAG.RAG_build.document_parser import EDADocumentParser
from src.EDA_RAG.RAG_build.vector_store import FAISSVectorStore
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class EDAVectorDBBuilder:
    """
    Build the complete EDA RAG vector database.

    Pipeline:
        EDADocumentLoader
                ↓
        EDADocumentParser
                ↓
        EDAChunker
                ↓
        EmbeddingService
                ↓
        FAISSVectorStore
    """

    def __init__(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        loader: EDADocumentLoader | None = None,
        parser: EDADocumentParser | None = None,
        chunker: EDASemanticChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: FAISSVectorStore | None = None,
    ) -> None:
        """Initialize the EDA vector database builder."""
        if not dataset_id.strip():
            raise ValueError("dataset_id cannot be empty.")

        self.dataset_id = dataset_id.strip()
        self.loader = (
            loader if loader is not None else EDADocumentLoader(dataset_id=self.dataset_id)
        )
        self.parser = parser if parser is not None else EDADocumentParser()
        self.chunker = chunker if chunker is not None else EDASemanticChunker()
        self.embedding_service = (
            embedding_service if embedding_service is not None else EmbeddingService()
        )
        self.vector_store = (
            vector_store if vector_store is not None else FAISSVectorStore()
        )

        RAG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = get_log("EDAVectorDBBuilder", log_dir=RAG_LOG_DIR)
        self.logger.info(
            "EDA vector DB builder initialized: dataset_id=%s", self.dataset_id
        )

    @track_performance
    def build(self) -> dict[str, Any]:
        """
        Execute the complete EDA RAG vector DB construction.

        Returns
        -------
        dict[str, Any]
            Build statistics and persisted vector DB information.
        """
        try:
            self.logger.info("Starting EDA vector database construction.")

            # 1. LOAD DOCUMENTS
            documents = self.loader.load_documents()
            if not documents:
                raise ValueError("No EDA documents were loaded.")
            self.logger.info("Documents loaded: %d", len(documents))

            # 2. PARSE DOCUMENTS
            sections = self.parser.parse_documents(documents)
            if not sections:
                raise ValueError("No EDA sections were generated.")
            self.logger.info("EDA sections generated: %d", len(sections))

            # 3. CHUNK SECTIONS
            all_chunks = []
            for section in sections:
                self.logger.debug("Chunking section: %s", section.section_id)
                chunks = self.chunker.chunk_section(section)
                if not chunks:
                    self.logger.warning(
                        "No chunks generated for section: %s", section.section_id
                    )
                    continue
                all_chunks.extend(chunks)

            if not all_chunks:
                raise ValueError("No EDA chunks were generated.")
            self.logger.info("EDA chunks generated: %d", len(all_chunks))

            # 4. GENERATE EMBEDDINGS
            embeddings = self.embedding_service.embed_chunks(all_chunks)
            self.logger.info(
                "Embeddings generated: rows=%d | dimension=%d",
                embeddings.shape[0],
                embeddings.shape[1],
            )

            # 5. ADD TO FAISS
            self.vector_store.add(chunks=all_chunks, embeddings=embeddings)
            self.logger.info("Vectors added to FAISS: %d", self.vector_store.count)

            # 6. SAVE VECTOR DATABASE
            self.vector_store.save()
            self.logger.info("Vector database persisted successfully.")

            # 7. RETURN BUILD SUMMARY
            result = self._build_result(
                documents=documents,
                sections=sections,
                chunks=all_chunks,
                embedding_shape=embeddings.shape,
            )
            self.logger.info("EDA vector database construction completed.")
            return result
        except Exception as exc:
            self.logger.error("EDA vector database construction failed: %s", exc)
            raise CustomException(exc, self.logger) from exc

    def _build_result(
        self,
        documents: list[Any],
        sections: list[Any],
        chunks: list[Any],
        embedding_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        """Build a summary of the vector DB construction."""
        documents_by_type: dict[str, int] = {}
        sections_by_type: dict[str, int] = {}
        chunks_by_type: dict[str, int] = {}

        for document in documents:
            analysis_type = document.analysis_type
            documents_by_type[analysis_type] = documents_by_type.get(analysis_type, 0) + 1

        for section in sections:
            analysis_type = section.analysis_type
            sections_by_type[analysis_type] = sections_by_type.get(analysis_type, 0) + 1

        for chunk in chunks:
            analysis_type = chunk.analysis_type
            chunks_by_type[analysis_type] = chunks_by_type.get(analysis_type, 0) + 1

        return {
            "dataset_id": self.dataset_id,
            "documents": len(documents),
            "documents_by_analysis_type": documents_by_type,
            "sections": len(sections),
            "sections_by_analysis_type": sections_by_type,
            "chunks": len(chunks),
            "chunks_by_analysis_type": chunks_by_type,
            "embedding_rows": embedding_shape[0],
            "embedding_dimension": embedding_shape[1],
            "vector_count": self.vector_store.count,
            "vector_dimension": self.vector_store.dimension,
            "index_path": str(self.vector_store.index_path),
            "metadata_path": str(self.vector_store.metadata_path),
        }
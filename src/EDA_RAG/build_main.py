"""
Build the EDA RAG vector database.
"""
from src.EDA_RAG.RAG_build.build_vector_db import EDAVectorDBBuilder


def build_rag() -> None:
    """Build the EDA vector database."""
    builder = EDAVectorDBBuilder(dataset_id="DS001")
    result = builder.build()

    print("\nEDA Vector Database")
    print("===================")
    print(f"Dataset: {result['dataset_id']}")
    print(f"Documents: {result['documents']}")
    print(f"Chunks: {result['chunks']}")
    print(f"Univariate chunks: {result['chunks_by_analysis_type']['univariate']}")
    print(f"Bivariate chunks: {result['chunks_by_analysis_type']['bivariate']}")
    print(f"Multivariate chunks: {result['chunks_by_analysis_type']['multivariate']}")
    print(f"Embedding dimension: {result['embedding_dimension']}")
    print(f"Vector count: {result['vector_count']}")
    print(f"Index: {result['index_path']}")
    print(f"Metadata: {result['metadata_path']}")


if __name__ == "__main__":
    build_rag()
"""
Run EDA RAG inference.

Executes the complete EDA RAG pipeline:

1. Query expansion
2. Vector retrieval
3. Cross-encoder reranking
4. Grounded answer generation
"""

from src.EDA_RAG.RAG_inference.inference_generation import rag_inference


def run_rag(query: str) -> None:
    """Run the EDA RAG inference pipeline."""

    query = query.strip()

    if not query:
        print("Query cannot be empty.")
        return

    try:
        answer = rag_inference.run(
            query=query,
            analysis_type=None,
            dataset_id="DS001",
        )

        print("\nEDA RAG Answer")
        print("==============")
        print(f"Question: {query}")
        print(f"\nAnswer:\n{answer}")

        return answer

    except Exception as exc:
        print("\nEDA RAG inference failed.")
        print(f"Error: {exc}")

    


if __name__ == "__main__":
    run_rag()
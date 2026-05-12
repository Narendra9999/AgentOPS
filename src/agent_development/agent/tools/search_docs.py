"""Vector search retrieval for documentation context."""

import logging

logger = logging.getLogger(__name__)


def search_docs(
    query: str,
    index_name: str,
    columns: list = None,
    num_results: int = 5,
) -> list[dict]:
    """
    Search the vector search index for relevant documents.

    Args:
        query: Search query text
        index_name: Fully qualified index name (catalog.schema.index)
        columns: Columns to return from the index
        num_results: Number of results to return

    Returns:
        list of dicts with keys matching the requested columns
    """
    default_columns = columns or ["chunk_text", "url", "chunk_id"]

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        results = w.vector_search_indexes.query_index(
            index_name=index_name,
            query_text=query,
            columns=default_columns,
            num_results=num_results,
        )

        docs = []
        if hasattr(results, "result") and results.result and results.result.data_array:
            for row in results.result.data_array:
                doc = {}
                for i, col in enumerate(default_columns):
                    doc[col] = row[i] if i < len(row) else ""
                docs.append(doc)

        logger.info(f"Retrieved {len(docs)} docs for: {query[:50]}...")
        return docs

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

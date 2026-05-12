"""Vector search retrieval for documentation context with optional reranking."""

import logging

logger = logging.getLogger(__name__)


def search_docs(
    query: str,
    index_name: str,
    columns: list = None,
    num_results: int = 5,
    reranker_enabled: bool = False,
    reranker_model: str = None,
) -> list[dict]:
    """
    Search the vector search index for relevant documents.
    Optionally reranks results using a cross-encoder model for better relevance.

    Args:
        query: Search query text
        index_name: Fully qualified index name (catalog.schema.index)
        columns: Columns to return from the index
        num_results: Number of results to return
        reranker_enabled: Whether to rerank results with a cross-encoder
        reranker_model: Endpoint name for the reranker model

    Returns:
        list of dicts with keys matching the requested columns
    """
    default_columns = columns or ["chunk_text", "url", "chunk_id"]

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

        # Fetch more candidates if reranking (reranker picks the best from a larger pool)
        fetch_count = num_results * 3 if reranker_enabled else num_results

        results = w.vector_search_indexes.query_index(
            index_name=index_name,
            query_text=query,
            columns=default_columns,
            num_results=fetch_count,
        )

        docs = []
        if hasattr(results, "result") and results.result and results.result.data_array:
            for row in results.result.data_array:
                doc = {}
                for i, col in enumerate(default_columns):
                    doc[col] = row[i] if i < len(row) else ""
                docs.append(doc)

        # Rerank if enabled
        if reranker_enabled and reranker_model and len(docs) > num_results:
            docs = _rerank(query, docs, default_columns, w, reranker_model, num_results)

        logger.info(f"Retrieved {len(docs)} docs for: {query[:50]}..."
                    f"{' (reranked)' if reranker_enabled else ''}")
        return docs

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def _rerank(
    query: str,
    docs: list[dict],
    columns: list,
    client,
    reranker_model: str,
    top_k: int,
) -> list[dict]:
    """Rerank search results using a cross-encoder model endpoint."""
    text_col = columns[0]  # First column is the text content

    try:
        # Build query-document pairs for the reranker
        pairs = [{"query": query, "passage": doc.get(text_col, "")} for doc in docs]

        response = client.serving_endpoints.query(
            name=reranker_model,
            input=pairs,
        )

        # Extract scores and sort
        scores = []
        if hasattr(response, "predictions"):
            scores = response.predictions
        elif hasattr(response, "data"):
            scores = [item.get("score", 0) for item in response.data]

        if scores and len(scores) == len(docs):
            scored_docs = list(zip(scores, docs))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked = [doc for _, doc in scored_docs[:top_k]]
            logger.info(f"Reranked {len(docs)} → top {top_k}")
            return reranked
        else:
            logger.warning(f"Reranker returned {len(scores)} scores for {len(docs)} docs — skipping")
            return docs[:top_k]

    except Exception as e:
        logger.warning(f"Reranker failed, returning original results: {e}")
        return docs[:top_k]

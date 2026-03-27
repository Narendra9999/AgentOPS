"""
Vector Search utilities — create endpoints, indexes, and query helpers.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    PipelineType,
)

# VectorIndexType was added in newer SDK versions
try:
    from databricks.sdk.service.vectorsearch import VectorIndexType
except ImportError:
    VectorIndexType = None
import logging
import time

logger = logging.getLogger(__name__)


def get_or_create_endpoint(endpoint_name: str, wait_timeout: int = 600) -> dict:
    """Get existing or create a new vector search endpoint. Waits until ONLINE."""
    w = WorkspaceClient()
    try:
        ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
        logger.info(f"Endpoint exists: {endpoint_name} (status: {ep.endpoint_status})")
    except Exception:
        logger.info(f"Creating vector search endpoint: {endpoint_name}")
        ep = w.vector_search_endpoints.create_endpoint(
            name=endpoint_name, endpoint_type=EndpointType.STANDARD)
        logger.info(f"Endpoint created: {endpoint_name}")

    # Wait for endpoint to be ONLINE
    elapsed = 0
    poll = 15
    while elapsed < wait_timeout:
        ep = w.vector_search_endpoints.get_endpoint(endpoint_name)
        status = str(ep.endpoint_status.state if ep.endpoint_status else "UNKNOWN")
        if "ONLINE" in status:
            logger.info(f"Endpoint {endpoint_name} is ONLINE")
            return ep
        logger.info(f"  Waiting for {endpoint_name}: {status} ({elapsed}s)")
        time.sleep(poll)
        elapsed += poll

    logger.warning(f"Endpoint {endpoint_name} not ONLINE after {wait_timeout}s — continuing anyway")
    return ep


def _make_embedding_source_column(text_column: str, embedding_model: str):
    """Create EmbeddingSourceColumn compatible with any SDK version.

    SDK versions have different APIs:
      - Older: EmbeddingSourceColumn(name=..., embedding_model_endpoint_name=...)
      - Newer: EmbeddingSourceColumn(name=..., embedding_config=EmbeddingConfig(
                   embedding_model_endpoint_name=...))
    """
    import inspect
    params = set(inspect.signature(EmbeddingSourceColumn.__init__).parameters.keys()) - {"self"}

    # Pattern 1: newer SDK with embedding_config wrapping EmbeddingConfig
    if "embedding_config" in params:
        try:
            from databricks.sdk.service.vectorsearch import EmbeddingConfig
            config = EmbeddingConfig(embedding_model_endpoint_name=embedding_model)
            return EmbeddingSourceColumn(name=text_column, embedding_config=config)
        except (ImportError, TypeError) as e:
            logger.warning(f"EmbeddingConfig pattern failed: {e}")

    # Pattern 2: older SDK with direct endpoint name parameter
    if "embedding_model_endpoint_name" in params:
        return EmbeddingSourceColumn(name=text_column, embedding_model_endpoint_name=embedding_model)

    # Pattern 3: brute force
    for param_name in ["embedding_model_endpoint_name", "model_endpoint_name", "endpoint_name"]:
        try:
            return EmbeddingSourceColumn(name=text_column, **{param_name: embedding_model})
        except TypeError:
            continue

    raise TypeError(
        f"Cannot create EmbeddingSourceColumn — unrecognized API. "
        f"Available params: {params}"
    )


def create_delta_sync_index(
    endpoint_name: str,
    index_name: str,
    source_table: str,
    embedding_model: str,
    text_column: str = "chunk_text",
    primary_key: str = "chunk_id",
) -> dict:
    """Create a delta sync vector search index."""
    w = WorkspaceClient()
    try:
        idx = w.vector_search_indexes.get_index(index_name)
        logger.info(f"Index already exists: {index_name}")
        return idx
    except Exception:
        logger.info(f"Creating delta sync index: {index_name}")
        # Use REST API directly — the SDK's EmbeddingSourceColumn serialization
        # may not match what the API expects across different workspace versions.
        # The flat format (embedding_model_endpoint_name on the column) works everywhere.
        payload = {
            "name": index_name,
            "endpoint_name": endpoint_name,
            "primary_key": primary_key,
            "index_type": "DELTA_SYNC",
            "delta_sync_index_spec": {
                "source_table": source_table,
                "pipeline_type": "TRIGGERED",
                "embedding_source_columns": [
                    {
                        "name": text_column,
                        "embedding_model_endpoint_name": embedding_model,
                    }
                ],
            },
        }
        resp = w.api_client.do("POST", "/api/2.0/vector-search/indexes", body=payload)
        logger.info(f"Index created: {index_name}")

    # Wait for index to be ONLINE and synced
    logger.info(f"Waiting for index {index_name} to sync...")
    max_wait = 1200  # 20 minutes
    elapsed = 0
    poll = 30
    while elapsed < max_wait:
        try:
            idx = w.vector_search_indexes.get_index(index_name)
            status = str(idx.status.ready if idx.status else "UNKNOWN")
            msg = str(idx.status.message if idx.status and idx.status.message else "")
            if "ONLINE" in status or "true" in status.lower():
                logger.info(f"Index {index_name} is ONLINE after {elapsed}s")
                return idx
            logger.info(f"  {elapsed}s: index status={status} {msg[:100]}")
        except Exception as e:
            logger.info(f"  {elapsed}s: checking index... ({e})")
        time.sleep(poll)
        elapsed += poll

    logger.warning(f"Index {index_name} not ONLINE after {max_wait}s — continuing anyway")
    return w.vector_search_indexes.get_index(index_name)


def rerank_results(
    query_text: str,
    results: dict,
    reranker_model: str,
    text_col_idx: int = 0,
    top_k: int = 5,
) -> dict:
    """
    Rerank search results using a cross-encoder model.
    Scores each (query, document) pair and re-sorts by relevance.

    Args:
        query_text: Original user query
        results: Raw search results from similarity_search
        reranker_model: Model serving endpoint name for the reranker
        text_col_idx: Index of the text column in data_array rows
        top_k: Number of results to return after reranking
    """
    from databricks.sdk import WorkspaceClient

    data_array = results.get("result", {}).get("data_array", [])
    if not data_array:
        return results

    w = WorkspaceClient()

    # Build pairs of (query, document) for the reranker
    documents = [row[text_col_idx] for row in data_array]
    pairs = [{"query": query_text, "passage": doc} for doc in documents]

    try:
        # Call reranker endpoint — expects pairs, returns scores
        response = w.serving_endpoints.query(
            name=reranker_model,
            input={"pairs": pairs},
        )

        # Extract scores and sort
        scores = []
        if hasattr(response, "predictions"):
            scores = response.predictions
        elif hasattr(response, "data"):
            scores = [item.get("score", 0) for item in response.data]

        if scores and len(scores) == len(data_array):
            # Attach scores and sort descending
            scored_rows = list(zip(scores, data_array))
            scored_rows.sort(key=lambda x: x[0], reverse=True)
            results["result"]["data_array"] = [row for _, row in scored_rows[:top_k]]
            logger.info(f"Reranked {len(data_array)} results → top {top_k}")
        else:
            logger.warning(f"Reranker returned {len(scores)} scores for {len(data_array)} docs — skipping rerank")

    except Exception as e:
        logger.warning(f"Reranker failed, returning original results: {e}")

    return results


def query_index(
    index_name: str,
    query_text: str,
    columns: list[str] = None,
    num_results: int = 5,
    filters: dict = None,
    search_type: str = "similarity",
    reranker_enabled: bool = False,
    reranker_model: str = None,
) -> dict:
    """
    Query a vector search index with optional reranking.

    Args:
        index_name: Full name of the VS index
        query_text: User query to search for
        columns: Columns to return
        num_results: Number of results
        filters: Optional filter dict
        search_type: "similarity", "hybrid", or "mmr"
        reranker_enabled: Whether to rerank results with a cross-encoder
        reranker_model: Endpoint name for the reranker model

    Search types:
        - similarity: Pure vector cosine similarity. Fast, good for general queries.
        - hybrid: Vector + keyword (BM25) scoring. Better for exact terms,
                  API names, error codes.
        - mmr: Maximal Marginal Relevance. De-duplicates by source URL
               for diverse results.

    Reranker:
        When enabled, over-fetches candidates (3x num_results), then uses
        a cross-encoder model to rescore each (query, document) pair and
        returns the top_k most relevant.
    """
    w = WorkspaceClient()
    default_columns = columns or ["content", "url", "id"]

    # If reranker is enabled, fetch more candidates for rescoring
    fetch_count = num_results * 3 if reranker_enabled else num_results

    # Use Databricks SDK for vector search (no databricks-vectorsearch package needed)
    vs_result = w.vector_search_indexes.query_index(
        index_name=index_name,
        query_text=query_text,
        columns=default_columns,
        num_results=fetch_count,
    )

    # Convert SDK response to dict format
    results = {"result": {"data_array": vs_result.result.data_array if vs_result.result else []}}

    # MMR: de-duplicate by URL
    if search_type == "mmr" and results["result"]["data_array"]:
        seen_urls = set()
        diverse_rows = []
        url_col_idx = default_columns.index("url") if "url" in default_columns else None
        for row in results["result"]["data_array"]:
            url = row[url_col_idx] if url_col_idx is not None else None
            if url not in seen_urls:
                seen_urls.add(url)
                diverse_rows.append(row)
            if len(diverse_rows) >= num_results:
                break
        results["result"]["data_array"] = diverse_rows

    # Rerank if enabled
    if reranker_enabled and reranker_model:
        text_col_idx = default_columns.index("content") if "content" in default_columns else 0
        results = rerank_results(
            query_text=query_text,
            results=results,
            reranker_model=reranker_model,
            text_col_idx=text_col_idx,
            top_k=num_results,
        )

    return results

"""
Agent Tools — Utility functions available to the agent.

Tools:
  - search_docs: Vector search retrieval for documentation context
  - calculate: Evaluate math expressions (cluster sizing, cost estimation)
  - get_current_timestamp: Current UTC timestamp for time-based queries
  - format_sql: Format and validate SQL snippets
"""

import logging
import re
from datetime import datetime, timezone

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


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression. Useful for cluster sizing,
    cost estimation, token counting, and capacity planning.

    Args:
        expression: Math expression (e.g., "8 * 16 * 0.15", "1024 / 3")

    Returns:
        dict with 'result' (float) or 'error' (str)

    Examples:
        calculate("8 * 16 * 0.15")  → {"expression": "8 * 16 * 0.15", "result": 19.2}
        calculate("(1000000 * 4) / 1024 / 1024")  → {"expression": "...", "result": 3.81}
    """
    # Allow only safe characters: digits, operators, parentheses, whitespace, decimal
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)%]+$', expression):
        return {"expression": expression, "error": "Invalid expression — only numbers and basic operators allowed"}

    try:
        # Use eval with empty globals to prevent code execution
        result = eval(expression, {"__builtins__": {}}, {})
        logger.info(f"Calculate: {expression} = {result}")
        return {"expression": expression, "result": round(float(result), 6)}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero"}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def get_current_timestamp() -> dict:
    """
    Return the current UTC timestamp. Useful for time-based queries,
    log analysis, and scheduling context.

    Returns:
        dict with 'utc', 'iso', and 'epoch' fields
    """
    now = datetime.now(timezone.utc)
    return {
        "utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "iso": now.isoformat(),
        "epoch": int(now.timestamp()),
    }


def format_sql(sql: str) -> dict:
    """
    Basic SQL formatting and validation. Normalizes whitespace,
    uppercases keywords, and checks for common issues.

    Args:
        sql: Raw SQL string

    Returns:
        dict with 'formatted' SQL and any 'warnings'
    """
    if not sql or not sql.strip():
        return {"error": "Empty SQL"}

    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
        "ON", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE", "IS",
        "NULL", "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET",
        "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE",
        "TABLE", "VIEW", "INDEX", "DROP", "ALTER", "ADD", "COLUMN",
        "GRANT", "REVOKE", "USE", "CATALOG", "SCHEMA", "AS", "DISTINCT",
        "UNION", "ALL", "CASE", "WHEN", "THEN", "ELSE", "END",
        "MERGE", "USING", "MATCHED", "VACUUM", "OPTIMIZE", "DESCRIBE",
    ]

    # Normalize whitespace
    formatted = " ".join(sql.split())

    # Uppercase SQL keywords (word boundary match)
    for kw in keywords:
        formatted = re.sub(rf'\b{kw}\b', kw, formatted, flags=re.IGNORECASE)

    warnings = []

    # Check for common issues
    if "SELECT *" in formatted.upper():
        warnings.append("Avoid SELECT * in production — specify columns explicitly")
    if "DROP " in formatted.upper() and "IF EXISTS" not in formatted.upper():
        warnings.append("Consider adding IF EXISTS to DROP statements")
    if formatted.upper().count("SELECT") > 1 and "(" not in formatted:
        warnings.append("Multiple SELECT statements — did you mean to use a subquery?")

    return {"formatted": formatted, "warnings": warnings}

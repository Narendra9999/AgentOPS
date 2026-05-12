"""
Agent Tools — Utility functions available to the agent.

Tools:
  - search_docs: Vector search retrieval for documentation context
  - calculate: Evaluate math expressions (cluster sizing, cost estimation)
  - get_current_timestamp: Current UTC timestamp for time-based queries
  - format_sql: Format and validate SQL snippets
  - cluster_sizing: Recommend cluster config based on dataset size and use case
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


# ── Node type catalog (common Databricks instance types) ──────
NODE_CATALOG = {
    # AWS instance types
    "i3.xlarge":    {"cloud": "AWS", "vcpus": 4,  "memory_gb": 30.5,  "storage_gb": 950,   "category": "storage_optimized", "dbu_per_hour": 1.0},
    "i3.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 61,    "storage_gb": 1900,  "category": "storage_optimized", "dbu_per_hour": 2.0},
    "i3.4xlarge":   {"cloud": "AWS", "vcpus": 16, "memory_gb": 122,   "storage_gb": 3800,  "category": "storage_optimized", "dbu_per_hour": 4.0},
    "m5.xlarge":    {"cloud": "AWS", "vcpus": 4,  "memory_gb": 16,    "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 0.75},
    "m5.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 32,    "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 1.5},
    "m5.4xlarge":   {"cloud": "AWS", "vcpus": 16, "memory_gb": 64,    "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 3.0},
    "m5.8xlarge":   {"cloud": "AWS", "vcpus": 32, "memory_gb": 128,   "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 6.0},
    "m5.12xlarge":  {"cloud": "AWS", "vcpus": 48, "memory_gb": 192,   "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 9.0},
    "m5.16xlarge":  {"cloud": "AWS", "vcpus": 64, "memory_gb": 256,   "storage_gb": 0,     "category": "general_purpose",   "dbu_per_hour": 12.0},
    "r5.xlarge":    {"cloud": "AWS", "vcpus": 4,  "memory_gb": 32,    "storage_gb": 0,     "category": "memory_optimized",  "dbu_per_hour": 1.0},
    "r5.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 64,    "storage_gb": 0,     "category": "memory_optimized",  "dbu_per_hour": 2.0},
    "r5.4xlarge":   {"cloud": "AWS", "vcpus": 16, "memory_gb": 128,   "storage_gb": 0,     "category": "memory_optimized",  "dbu_per_hour": 4.0},
    "r5.8xlarge":   {"cloud": "AWS", "vcpus": 32, "memory_gb": 256,   "storage_gb": 0,     "category": "memory_optimized",  "dbu_per_hour": 8.0},
    "c5.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 16,    "storage_gb": 0,     "category": "compute_optimized", "dbu_per_hour": 1.5},
    "c5.4xlarge":   {"cloud": "AWS", "vcpus": 16, "memory_gb": 32,    "storage_gb": 0,     "category": "compute_optimized", "dbu_per_hour": 3.0},
    "c5.9xlarge":   {"cloud": "AWS", "vcpus": 36, "memory_gb": 72,    "storage_gb": 0,     "category": "compute_optimized", "dbu_per_hour": 6.75},
    "p3.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 61,    "storage_gb": 0,     "category": "gpu",               "dbu_per_hour": 5.5, "gpus": 1, "gpu_type": "V100"},
    "p3.8xlarge":   {"cloud": "AWS", "vcpus": 32, "memory_gb": 244,   "storage_gb": 0,     "category": "gpu",               "dbu_per_hour": 22.0, "gpus": 4, "gpu_type": "V100"},
    "g5.xlarge":    {"cloud": "AWS", "vcpus": 4,  "memory_gb": 16,    "storage_gb": 250,   "category": "gpu",               "dbu_per_hour": 2.0, "gpus": 1, "gpu_type": "A10G"},
    "g5.2xlarge":   {"cloud": "AWS", "vcpus": 8,  "memory_gb": 32,    "storage_gb": 450,   "category": "gpu",               "dbu_per_hour": 4.0, "gpus": 1, "gpu_type": "A10G"},
    # Azure instance types
    "Standard_DS3_v2":  {"cloud": "Azure", "vcpus": 4,  "memory_gb": 14,  "storage_gb": 28,  "category": "general_purpose",   "dbu_per_hour": 0.75},
    "Standard_DS4_v2":  {"cloud": "Azure", "vcpus": 8,  "memory_gb": 28,  "storage_gb": 56,  "category": "general_purpose",   "dbu_per_hour": 1.5},
    "Standard_DS5_v2":  {"cloud": "Azure", "vcpus": 16, "memory_gb": 56,  "storage_gb": 112, "category": "general_purpose",   "dbu_per_hour": 3.0},
    "Standard_E8s_v3":  {"cloud": "Azure", "vcpus": 8,  "memory_gb": 64,  "storage_gb": 128, "category": "memory_optimized",  "dbu_per_hour": 2.0},
    "Standard_E16s_v3": {"cloud": "Azure", "vcpus": 16, "memory_gb": 128, "storage_gb": 256, "category": "memory_optimized",  "dbu_per_hour": 4.0},
    "Standard_F8s_v2":  {"cloud": "Azure", "vcpus": 8,  "memory_gb": 16,  "storage_gb": 64,  "category": "compute_optimized", "dbu_per_hour": 1.5},
    "Standard_NC6s_v3": {"cloud": "Azure", "vcpus": 6,  "memory_gb": 112, "storage_gb": 736, "category": "gpu",               "dbu_per_hour": 8.0, "gpus": 1, "gpu_type": "V100"},
}

# Use case profiles — recommended category + memory/core ratios
USE_CASE_PROFILES = {
    "etl": {
        "description": "ETL / Data Engineering",
        "recommended_category": "general_purpose",
        "memory_per_gb_data": 2,        # GB RAM per GB of data
        "cores_per_gb_data": 0.5,       # vCPUs per GB of data
        "min_nodes": 2,
        "max_nodes": 20,
        "tips": [
            "Use auto-scaling (min 2, max based on data size)",
            "Enable spark.databricks.delta.optimizeWrite for Delta writes",
            "Consider i3 instances for shuffle-heavy workloads",
        ],
    },
    "ml_training": {
        "description": "ML Model Training",
        "recommended_category": "memory_optimized",
        "memory_per_gb_data": 4,
        "cores_per_gb_data": 1,
        "min_nodes": 1,
        "max_nodes": 10,
        "tips": [
            "Use memory-optimized instances (r5 on AWS, E-series on Azure)",
            "For deep learning, use GPU instances (p3/g5 on AWS, NC on Azure)",
            "Set spark.sql.shuffle.partitions based on data size",
        ],
    },
    "streaming": {
        "description": "Streaming / Real-time Processing",
        "recommended_category": "compute_optimized",
        "memory_per_gb_data": 1.5,
        "cores_per_gb_data": 1,
        "min_nodes": 2,
        "max_nodes": 10,
        "tips": [
            "Use compute-optimized for low-latency processing",
            "Set min workers >= 2 for availability",
            "Enable auto-scaling with tight bounds",
        ],
    },
    "sql_analytics": {
        "description": "SQL Analytics / BI Queries",
        "recommended_category": "general_purpose",
        "memory_per_gb_data": 1,
        "cores_per_gb_data": 0.25,
        "min_nodes": 1,
        "max_nodes": 8,
        "tips": [
            "Consider SQL Warehouses with Photon for best SQL performance",
            "Use Delta caching (spark.databricks.io.cache.enabled)",
            "Liquid clustering for large frequently-queried tables",
        ],
    },
    "ml_inference": {
        "description": "ML Model Inference / Batch Scoring",
        "recommended_category": "compute_optimized",
        "memory_per_gb_data": 1,
        "cores_per_gb_data": 0.5,
        "min_nodes": 2,
        "max_nodes": 10,
        "tips": [
            "Use model serving endpoints for real-time inference",
            "For batch: partition input data and use mapInPandas",
            "GPU instances for deep learning inference (g5 on AWS)",
        ],
    },
}


def get_node_info(node_type: str = None) -> dict:
    """
    Get specs for a specific node type or list all available node types.

    Args:
        node_type: Instance type name (e.g., "m5.2xlarge"). None = list all.

    Returns:
        dict with node specs or list of all available types
    """
    if node_type:
        node_type = node_type.strip()
        if node_type in NODE_CATALOG:
            info = NODE_CATALOG[node_type].copy()
            info["node_type"] = node_type
            return info
        # Fuzzy match
        matches = [k for k in NODE_CATALOG if node_type.lower() in k.lower()]
        if matches:
            return {"error": f"Node type '{node_type}' not found. Did you mean: {', '.join(matches)}?"}
        return {"error": f"Node type '{node_type}' not found. Use get_node_info() to list all types."}

    # List all grouped by category
    by_category = {}
    for name, specs in NODE_CATALOG.items():
        cat = specs["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "node_type": name,
            "vcpus": specs["vcpus"],
            "memory_gb": specs["memory_gb"],
            "cloud": specs["cloud"],
        })
    return {"node_types": by_category, "total": len(NODE_CATALOG)}


def cluster_sizing(
    data_size_gb: float,
    use_case: str = "etl",
    node_type: str = None,
    cloud: str = "AWS",
) -> dict:
    """
    Recommend cluster configuration based on dataset size and use case.

    Args:
        data_size_gb: Size of the dataset in GB
        use_case: One of "etl", "ml_training", "streaming", "sql_analytics", "ml_inference"
        node_type: Preferred node type (optional — auto-selects if not specified)
        cloud: "AWS" or "Azure"

    Returns:
        dict with recommended cluster config, node specs, and sizing breakdown
    """
    use_case = use_case.lower().replace(" ", "_").replace("-", "_")
    if use_case not in USE_CASE_PROFILES:
        return {
            "error": f"Unknown use case '{use_case}'",
            "available_use_cases": list(USE_CASE_PROFILES.keys()),
        }

    profile = USE_CASE_PROFILES[use_case]

    # Select node type
    if node_type and node_type in NODE_CATALOG:
        selected_node = node_type
        node_specs = NODE_CATALOG[node_type]
    else:
        # Auto-select: pick the best node for this use case and cloud
        candidates = [
            (name, specs) for name, specs in NODE_CATALOG.items()
            if specs["category"] == profile["recommended_category"]
            and specs["cloud"] == cloud
            and specs.get("gpus", 0) == 0  # skip GPU unless ML
        ]
        if use_case in ("ml_training",) and cloud == "AWS":
            gpu_candidates = [
                (name, specs) for name, specs in NODE_CATALOG.items()
                if specs["category"] == "gpu" and specs["cloud"] == cloud
            ]
            if gpu_candidates:
                candidates = gpu_candidates

        if not candidates:
            candidates = [
                (name, specs) for name, specs in NODE_CATALOG.items()
                if specs["cloud"] == cloud and specs.get("gpus", 0) == 0
            ]

        # Pick mid-range option
        candidates.sort(key=lambda x: x[1]["vcpus"])
        idx = min(len(candidates) // 2, len(candidates) - 1)
        selected_node, node_specs = candidates[idx]

    # Calculate required resources
    required_memory_gb = data_size_gb * profile["memory_per_gb_data"]
    required_cores = data_size_gb * profile["cores_per_gb_data"]

    # Calculate number of nodes
    nodes_by_memory = max(1, int(required_memory_gb / node_specs["memory_gb"]) + 1)
    nodes_by_cores = max(1, int(required_cores / node_specs["vcpus"]) + 1)
    recommended_nodes = max(nodes_by_memory, nodes_by_cores)
    recommended_nodes = max(profile["min_nodes"], min(recommended_nodes, profile["max_nodes"]))

    # Total cluster capacity
    total_cores = recommended_nodes * node_specs["vcpus"]
    total_memory_gb = recommended_nodes * node_specs["memory_gb"]
    total_dbu_per_hour = recommended_nodes * node_specs["dbu_per_hour"]

    # Spark partitions recommendation
    recommended_partitions = max(total_cores * 2, 200)

    return {
        "use_case": profile["description"],
        "data_size_gb": data_size_gb,
        "recommendation": {
            "node_type": selected_node,
            "num_workers": recommended_nodes,
            "driver_node_type": selected_node,
            "autoscale": {
                "min_workers": profile["min_nodes"],
                "max_workers": recommended_nodes,
            },
            "spark_config": {
                "spark.sql.shuffle.partitions": str(recommended_partitions),
            },
        },
        "cluster_capacity": {
            "total_vcpus": total_cores,
            "total_memory_gb": total_memory_gb,
            "total_storage_gb": recommended_nodes * node_specs.get("storage_gb", 0),
            "total_dbu_per_hour": round(total_dbu_per_hour, 2),
            "gpus": recommended_nodes * node_specs.get("gpus", 0) if node_specs.get("gpus") else None,
        },
        "node_specs": {
            "node_type": selected_node,
            "cloud": node_specs["cloud"],
            "vcpus": node_specs["vcpus"],
            "memory_gb": node_specs["memory_gb"],
            "storage_gb": node_specs.get("storage_gb", 0),
            "category": node_specs["category"],
        },
        "sizing_breakdown": {
            "required_memory_gb": round(required_memory_gb, 1),
            "required_cores": round(required_cores, 1),
            "nodes_by_memory": nodes_by_memory,
            "nodes_by_cores": nodes_by_cores,
            "limiting_factor": "memory" if nodes_by_memory >= nodes_by_cores else "compute",
        },
        "tips": profile["tips"],
    }

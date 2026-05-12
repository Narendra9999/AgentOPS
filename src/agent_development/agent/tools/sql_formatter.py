"""SQL formatting and validation utility."""

import re


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

    formatted = " ".join(sql.split())

    for kw in keywords:
        formatted = re.sub(rf'\b{kw}\b', kw, formatted, flags=re.IGNORECASE)

    warnings = []

    if "SELECT *" in formatted.upper():
        warnings.append("Avoid SELECT * in production — specify columns explicitly")
    if "DROP " in formatted.upper() and "IF EXISTS" not in formatted.upper():
        warnings.append("Consider adding IF EXISTS to DROP statements")
    if formatted.upper().count("SELECT") > 1 and "(" not in formatted:
        warnings.append("Multiple SELECT statements — did you mean to use a subquery?")

    return {"formatted": formatted, "warnings": warnings}

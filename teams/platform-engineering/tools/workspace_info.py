"""
Custom Tool: Workspace Info Lookup
Team: Platform Engineering

Returns workspace metadata and configuration details.
"""

TOOL_NAME = "workspace_info"
TOOL_DESCRIPTION = "Get current Databricks workspace information (region, URL, runtime version)"
TRIGGER_PATTERNS = [
    "workspace info",
    "which workspace",
    "workspace details",
    "workspace region",
    "workspace url",
    "what workspace am i on",
]


def execute(user_message: str = "", **kwargs) -> dict:
    """Return workspace metadata from the SDK."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()

        return {
            "workspace_host": w.config.host or "unknown",
            "workspace_id": getattr(w.config, "workspace_id", "unknown"),
            "cloud": _detect_cloud(w.config.host or ""),
        }
    except Exception as e:
        return {
            "error": str(e),
            "note": "Workspace info unavailable — SDK not configured in this environment",
        }


def _detect_cloud(host: str) -> str:
    if "azure" in host.lower():
        return "Azure"
    elif "gcp" in host.lower():
        return "GCP"
    return "AWS"

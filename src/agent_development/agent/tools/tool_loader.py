"""
Tool Plugin Loader — dynamically loads team-specific tools.

Team tools follow a standard interface:
    TOOL_NAME: str           — unique tool identifier
    TOOL_DESCRIPTION: str    — what the tool does (injected into system prompt)
    TRIGGER_PATTERNS: list   — keywords that trigger the tool
    execute(**kwargs) -> dict — the tool function

Team tools live in: teams/{team}/tools/*.py
They are packaged into the model via code_paths in RegisterModel.py.
"""

import importlib
import os
import sys
import logging

logger = logging.getLogger(__name__)


def load_custom_tools(tool_names: list, tools_dir: str = None) -> dict:
    """
    Load custom tools by name from the custom_tools directory.

    Args:
        tool_names: List of tool module names to load (e.g., ["query_cluster_status"])
        tools_dir: Directory containing custom tool .py files. If None, looks for
                   'custom_tools/' relative to this file's directory.

    Returns:
        dict of {tool_name: tool_module} for successfully loaded tools
    """
    if not tool_names:
        return {}

    # Resolve tools directory — check multiple locations
    # When packaged by MLflow, code_paths are extracted to the model directory
    if tools_dir is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "custom_tools"),  # local dev
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "custom_tools"),  # model serving
        ]
        # Also check all sys.path entries for the tool modules directly
        for path in candidates:
            if os.path.isdir(path):
                tools_dir = path
                break

    if tools_dir is None or not os.path.isdir(tools_dir):
        # Tools might be importable directly (code_paths adds to sys.path)
        logger.info("No custom_tools directory found — trying direct import")
        tools_dir = None

    # Add to Python path so imports work
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)

    loaded = {}
    for name in tool_names:
        try:
            module = importlib.import_module(name)

            # Validate standard interface
            if not hasattr(module, "execute"):
                logger.warning(f"Custom tool '{name}' missing execute() function — skipping")
                continue

            loaded[name] = {
                "name": getattr(module, "TOOL_NAME", name),
                "description": getattr(module, "TOOL_DESCRIPTION", ""),
                "triggers": getattr(module, "TRIGGER_PATTERNS", []),
                "execute": module.execute,
            }
            logger.info(f"Loaded custom tool: {name} "
                        f"(triggers: {loaded[name]['triggers']})")

        except Exception as e:
            logger.warning(f"Failed to load custom tool '{name}': {e}")

    return loaded


def execute_custom_tools(custom_tools: dict, user_message: str) -> list:
    """
    Run custom tools whose trigger patterns match the user message.

    Args:
        custom_tools: dict from load_custom_tools()
        user_message: the user's query

    Returns:
        list of tool result strings to inject into context
    """
    results = []
    msg_lower = user_message.lower()

    for name, tool in custom_tools.items():
        triggers = tool.get("triggers", [])
        if not any(t.lower() in msg_lower for t in triggers):
            continue

        try:
            result = tool["execute"](user_message=user_message)
            if isinstance(result, dict):
                # Format dict as key-value output
                parts = [f"[{tool['name']}]"]
                for k, v in result.items():
                    parts.append(f"  {k}: {v}")
                results.append("\n".join(parts))
            elif isinstance(result, str):
                results.append(f"[{tool['name']}] {result}")
            else:
                results.append(f"[{tool['name']}] {str(result)}")

            logger.info(f"Custom tool executed: {name}")
        except Exception as e:
            logger.warning(f"Custom tool '{name}' failed: {e}")

    return results

"""
Example Custom Tool — copy and modify for your team.

Standard interface:
  TOOL_NAME: str           — unique identifier
  TOOL_DESCRIPTION: str    — injected into system prompt
  TRIGGER_PATTERNS: list   — keywords that trigger this tool
  execute(**kwargs) -> dict — the tool function
"""

TOOL_NAME = "example_tool"
TOOL_DESCRIPTION = "Example tool — replace with your tool's description"
TRIGGER_PATTERNS = ["example trigger", "sample query"]


def execute(user_message: str = "", **kwargs) -> dict:
    """Replace this with your tool logic."""
    return {
        "message": "This is an example tool response",
        "input": user_message[:100],
    }

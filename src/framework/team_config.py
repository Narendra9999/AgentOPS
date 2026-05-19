"""Team config loader.

Resolves team-specific deploy values from ``src/teams/<team_name>/config.yaml``.
Each pipeline notebook calls ``load_team_settings(team_name)`` to get a dict of
resolved values (agent_name, chatbot_name, schema, vs_index, ...). Empty
``team_name`` returns an empty dict, letting callers fall back to their own
defaults.
"""

from __future__ import annotations

import os
from typing import Any

import yaml


def _derive_chatbot(agent_name: str) -> str:
    return agent_name.replace("_", "-") + "-chatbot"


def _find_team_config_path(team_name: str, bundle_root: str | None = None) -> str:
    """Resolve ``src/teams/<team_name>/config.yaml`` under the bundle root.

    ``bundle_root`` defaults to the repo root (two levels above this file).
    """
    if bundle_root is None:
        # src/framework/team_config.py → bundle root is two levels up
        bundle_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(bundle_root, "src", "teams", team_name, "config.yaml")


def load_team_settings(
    team_name: str,
    bundle_root: str | None = None,
) -> dict[str, Any]:
    """Read team config and return a dict of resolved deploy values.

    Returns ``{}`` when ``team_name`` is empty so callers can keep their
    existing widget defaults.

    Keys returned (when team config is found):
        agent_name, chatbot_name, schema, audit_schema,
        vs_index, vs_endpoint, llm_endpoint, embedding_model,
        team_dir, team_config_path

    Naming precedence:
        - ``naming.schema`` || ``agent.name``
        - ``naming.audit_schema`` || ``agent.name + "_audit"``
        - ``naming.chatbot`` || derived ``agent.name.replace("_", "-") + "-chatbot"``
    """
    if not team_name:
        return {}

    path = _find_team_config_path(team_name, bundle_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Team config not found: {path}. "
            f"Check team_name='{team_name}' and that src/teams/{team_name}/config.yaml exists."
        )

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    agent_block = cfg.get("agent") or {}
    agent_name = agent_block.get("name")
    if not agent_name:
        raise ValueError(f"{path} is missing required field agent.name")

    naming = cfg.get("naming") or {}
    vs = cfg.get("vector_search") or {}
    llm = cfg.get("llm") or {}

    return {
        "agent_name": agent_name,
        "chatbot_name": naming.get("chatbot") or _derive_chatbot(agent_name),
        "schema": naming.get("schema") or agent_name,
        "audit_schema": naming.get("audit_schema") or f"{agent_name}_audit",
        "vs_index": vs.get("index", ""),
        "vs_endpoint": vs.get("endpoint", ""),
        "llm_endpoint": llm.get("endpoint", ""),
        "embedding_model": vs.get("embedding_model", ""),
        "team_dir": team_name,
        "team_config_path": path,
    }


def merge_with_overrides(
    settings: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge resolved team settings with widget/var overrides.

    Override semantics: a non-empty override wins; an empty/None override
    falls back to the team setting.
    """
    merged = dict(settings)
    for key, val in overrides.items():
        if val:
            merged[key] = val
    return merged

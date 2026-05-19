"""Unit tests for framework.team_config.load_team_settings."""

import os
import tempfile

import pytest
import yaml

from framework.team_config import (
    _derive_chatbot,
    load_team_settings,
    merge_with_overrides,
)


def _write_team_config(bundle_root: str, team_name: str, cfg: dict) -> str:
    team_dir = os.path.join(bundle_root, "src", "teams", team_name)
    os.makedirs(team_dir, exist_ok=True)
    path = os.path.join(team_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_empty_team_name_returns_empty_dict():
    assert load_team_settings("") == {}


def test_missing_team_raises():
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(FileNotFoundError, match="src/teams/missing"):
            load_team_settings("missing", bundle_root=root)


def test_naming_block_wins_over_derivation():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-a", {
            "agent": {"name": "team_a_agent"},
            "naming": {
                "schema": "team_a_schema",
                "audit_schema": "team_a_audit_custom",
                "chatbot": "team-a-bot",
            },
            "vector_search": {"index": "team_a_idx", "endpoint": "team-a-vs"},
            "llm": {"endpoint": "databricks-gpt-oss-120b"},
        })
        s = load_team_settings("team-a", bundle_root=root)
        assert s["agent_name"] == "team_a_agent"
        assert s["schema"] == "team_a_schema"
        assert s["audit_schema"] == "team_a_audit_custom"
        assert s["chatbot_name"] == "team-a-bot"
        assert s["vs_index"] == "team_a_idx"
        assert s["vs_endpoint"] == "team-a-vs"
        assert s["llm_endpoint"] == "databricks-gpt-oss-120b"
        assert s["team_dir"] == "team-a"


def test_naming_falls_back_to_derivation():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-b", {
            "agent": {"name": "team_b_agent"},
            "vector_search": {"index": "team_b_idx"},
        })
        s = load_team_settings("team-b", bundle_root=root)
        assert s["agent_name"] == "team_b_agent"
        assert s["schema"] == "team_b_agent"
        assert s["audit_schema"] == "team_b_agent_audit"
        assert s["chatbot_name"] == "team-b-agent-chatbot"


def test_missing_agent_name_raises():
    with tempfile.TemporaryDirectory() as root:
        _write_team_config(root, "team-c", {"vector_search": {"index": "x"}})
        with pytest.raises(ValueError, match="agent.name"):
            load_team_settings("team-c", bundle_root=root)


def test_derive_chatbot_helper():
    assert _derive_chatbot("platform_eng_docs_agent") == "platform-eng-docs-agent-chatbot"
    assert _derive_chatbot("agent") == "agent-chatbot"


def test_merge_with_overrides_non_empty_wins():
    settings = {"agent_name": "team_a", "schema": "team_a_schema", "vs_index": "team_a_idx"}
    overrides = {"agent_name": "override", "schema": "", "vs_index": None}
    merged = merge_with_overrides(settings, overrides)
    assert merged["agent_name"] == "override"
    assert merged["schema"] == "team_a_schema"
    assert merged["vs_index"] == "team_a_idx"

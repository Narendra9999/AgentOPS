"""
Unit tests for Post-LLM Guardrails.
Run: python -m pytest tests/unit/test_post_llm_guardrails.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'agentops_demo'))

from framework.guardrails.post_llm import PostLLMGuardrails


@pytest.fixture
def guardrails():
    return PostLLMGuardrails({
        "enabled_checks": ["toxicity", "compliance", "pii_leakage", "hallucination", "quality"],
        "compliance_blocked_phrases": [
            "this is the only way",
            "always use this approach",
            "guaranteed to work",
        ],
        "min_response_length": 20,
    })


# ── Toxicity ─────────────────────────────────────────────────

class TestResponseToxicity:
    def test_clean_response_passes(self, guardrails):
        result = guardrails.check(
            "What is Delta?",
            "Delta Lake is an open-source storage layer that brings ACID transactions to Apache Spark.",
            None)
        assert result["blocked"] is False

    def test_toxic_response_blocked(self, guardrails):
        result = guardrails.check(
            "What is Delta?",
            "You're stupid for not knowing this. I hate answering basic questions.",
            None)
        assert result["blocked"] is True
        assert result["blocked_by"] == "toxicity"


# ── Compliance ───────────────────────────────────────────────

class TestCompliance:
    def test_guaranteed_blocked(self, guardrails):
        result = guardrails.check(
            "Which approach should I use?",
            "This is the only way to do it. Always use this approach.",
            None)
        assert result["blocked"] is True
        assert result["blocked_by"] == "compliance"

    def test_normal_advice_passes(self, guardrails):
        result = guardrails.check(
            "Which approach?",
            "There are several approaches. For most use cases, Delta Lake merge is recommended.",
            None)
        assert result["blocked"] is False


# ── PII leakage ──────────────────────────────────────────────

class TestPIILeakage:
    def test_ssn_in_response_blocked(self, guardrails):
        result = guardrails.check(
            "Tell me about the author",
            "The author's SSN is 123-45-6789 and they work at Databricks.",
            None)
        assert result["blocked"] is True
        assert result["blocked_by"] == "pii_leakage"

    def test_credit_card_in_response_blocked(self, guardrails):
        result = guardrails.check(
            "Payment info",
            "Use credit card 4111-1111-1111-1111 for billing.",
            None)
        assert result["blocked"] is True

    def test_no_pii_passes(self, guardrails):
        result = guardrails.check(
            "How to create a table?",
            "Use CREATE TABLE catalog.schema.table (col1 STRING, col2 INT) to create a Delta table.",
            None)
        assert result["blocked"] is False


# ── Hallucination ────────────────────────────────────────────

class TestHallucination:
    def test_no_context_passes(self, guardrails):
        result = guardrails.check("question", "Some long response about Delta Lake and things.", None)
        assert result["blocked"] is False

    def test_grounded_urls_pass(self, guardrails):
        context = {"retrieved_docs": [
            {"url": "https://docs.databricks.com/en/delta/index.html", "content": "Delta Lake..."}
        ]}
        result = guardrails.check(
            "What is Delta?",
            "Delta Lake provides ACID transactions. See https://docs.databricks.com/en/delta/index.html",
            context)
        assert result["blocked"] is False

    def test_ungrounded_url_flagged(self, guardrails):
        context = {"retrieved_docs": [
            {"url": "https://docs.databricks.com/en/delta/index.html", "content": "Delta..."}
        ]}
        result = guardrails.check(
            "What is Delta?",
            "See https://docs.databricks.com/en/fake-page-that-doesnt-exist.html for details.",
            context)
        assert result["checks"]["hallucination"].get("flagged") is True


# ── Quality ──────────────────────────────────────────────────

class TestQuality:
    def test_short_response_flagged(self, guardrails):
        result = guardrails.check("What is Delta?", "It's a table.", None)
        assert result["checks"]["quality"].get("flagged") is True

    def test_good_response_passes(self, guardrails):
        result = guardrails.check(
            "What is Delta?",
            "Delta Lake is an open-source storage layer that brings ACID transactions, "
            "scalable metadata handling, and unifies streaming and batch data processing.",
            None)
        assert result["blocked"] is False

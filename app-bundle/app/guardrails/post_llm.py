"""
AgentOPS Framework — Post-LLM Guardrails
Runs AFTER the LLM generates a response, BEFORE returning to user.
Checks: toxicity, compliance, pii_leakage, hallucination, quality.
"""

import mlflow
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

RESPONSE_PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}

DEFAULT_COMPLIANCE_PHRASES = [
    "guaranteed return", "guaranteed profit", "you will make money",
    "risk-free investment", "insider information",
]


class PostLLMGuardrails:
    """
    Post-LLM guardrail pipeline.
    Configurable via config dict — each check can be enabled/disabled.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.enabled_checks = config.get("enabled_checks", [
            "toxicity", "compliance", "pii_leakage", "hallucination", "quality",
        ])
        self.compliance_blocked_phrases = config.get(
            "compliance_blocked_phrases", DEFAULT_COMPLIANCE_PHRASES)
        self.hallucination_threshold = config.get("hallucination_threshold", 0.5)
        self.min_response_length = config.get("min_response_length", 20)

    @mlflow.trace(span_type="GUARDRAIL", name="post_llm_check")
    def check(
        self,
        user_message: str,
        response: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Run all enabled post-LLM guardrail checks in order."""
        results = {"blocked": False, "checks": {}, "modifications": []}

        for check_name in self.enabled_checks:
            check_fn = getattr(self, f"_check_{check_name}", None)
            if check_fn:
                result = check_fn(user_message, response, context)
                results["checks"][check_name] = result
                if result.get("blocked"):
                    results["blocked"] = True
                    results["message"] = result.get("message", "Response filtered.")
                    results["blocked_by"] = check_name
                    logger.warning(f"Post-LLM '{check_name}' blocked: {result.get('reason', '')}")
                    break

        return results

    # ── Check: Response toxicity ───────────────────────────────

    def _check_toxicity(self, user_message: str, response: str, context) -> dict:
        toxic_patterns = [
            r"I hate",
            r"you('re| are) (stupid|dumb|idiot)",
            r"(racial|ethnic) (slur|epithet)",
            r"shut up",
        ]
        response_lower = response.lower()
        for pattern in toxic_patterns:
            if re.search(pattern, response_lower):
                return {
                    "blocked": True,
                    "reason": f"Toxic content in response: {pattern}",
                    "message": "I apologize, but I'm unable to provide that response. Let me help you differently.",
                }
        return {"blocked": False}

    # ── Check: Compliance violations ───────────────────────────

    def _check_compliance(self, user_message: str, response: str, context) -> dict:
        response_lower = response.lower()
        for phrase in self.compliance_blocked_phrases:
            if phrase in response_lower:
                return {
                    "blocked": True,
                    "reason": f"Compliance violation: '{phrase}'",
                    "message": "I need to rephrase my response to comply with communication guidelines.",
                }
        return {"blocked": False}

    # ── Check: PII leakage in response ─────────────────────────

    def _check_pii_leakage(self, user_message: str, response: str, context) -> dict:
        # Collect emails/identifiers to whitelist (user's own info, example addresses)
        whitelist = {"user@example.com", "example@databricks.com", "your_email@company.com"}

        # Domains that are safe in documentation context (not real PII)
        safe_email_domains = {
            "example.com", "example.org", "example.net",
            "databricks.com", "company.com", "domain.com",
            "mycompany.com", "test.com", "localhost",
        }

        if isinstance(context, dict):
            uid = context.get("user_id", "")
            if uid:
                whitelist.add(uid)

            # Whitelist emails found in retrieved documentation (not leakage)
            for doc in context.get("retrieved_docs", []):
                doc_text = doc.get("chunk_text", "") if isinstance(doc, dict) else ""
                for email_match in re.findall(RESPONSE_PII_PATTERNS["email"], doc_text):
                    whitelist.add(email_match)

        for pii_type, pattern in RESPONSE_PII_PATTERNS.items():
            matches = re.findall(pattern, response)
            # Filter out whitelisted values
            flagged = [m for m in matches if m not in whitelist]

            # For emails: also filter out safe documentation domains
            if pii_type == "email":
                flagged = [
                    m for m in flagged
                    if not any(m.lower().endswith(f"@{d}") for d in safe_email_domains)
                ]

            if flagged:
                return {
                    "blocked": True,
                    "reason": f"PII leakage in response: {pii_type} ({flagged[0][:20]}...)",
                    "message": "I detected potentially sensitive information in my response. Let me provide the information without personal data.",
                }
        return {"blocked": False}

    # ── Check: Hallucination (basic) ───────────────────────────

    def _check_hallucination(self, user_message: str, response: str, context) -> dict:
        """
        Basic hallucination check — verifies response claims exist in context.
        For production, use MLflow RetrievalGroundedness scorer.
        """
        # context may be a dict or an MLflow ChatContext pydantic object.
        # Safely extract retrieved_docs regardless of type.
        retrieved_docs = None
        if isinstance(context, dict):
            retrieved_docs = context.get("retrieved_docs")
        elif context is not None:
            retrieved_docs = getattr(context, "retrieved_docs", None)

        if not retrieved_docs:
            return {"blocked": False, "note": "No context to check against"}

        # Check if specific URL/page references in response exist in context
        retrieved_urls = set()
        for doc in retrieved_docs:
            url = doc.get("url", "") if isinstance(doc, dict) else getattr(doc, "url", "")
            if url:
                retrieved_urls.add(url)

        # Look for URL patterns in response that don't match retrieved docs
        url_pattern = r"https?://docs\.databricks\.com/[^\s\)]+"
        response_urls = re.findall(url_pattern, response)
        ungrounded_urls = [u for u in response_urls if u not in retrieved_urls]

        if ungrounded_urls:
            return {
                "blocked": False,
                "flagged": True,
                "reason": f"Potentially ungrounded URLs: {ungrounded_urls}",
            }

        return {"blocked": False}

    # ── Check: Response quality ────────────────────────────────

    def _check_quality(self, user_message: str, response: str, context) -> dict:
        if len(response.strip()) < self.min_response_length:
            return {
                "blocked": False,
                "flagged": True,
                "reason": f"Response too short ({len(response.strip())} chars)",
            }
        return {"blocked": False}

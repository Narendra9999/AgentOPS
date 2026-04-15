"""
AgentOPS Framework — Pre-LLM Guardrails
Runs BEFORE user input reaches the LLM.
Checks: input_length_min, input_length_max, pii, injection, toxicity, intent.

Note: LLM-based safety (toxicity, harmful content) is handled by AI Gateway
safety filter on the serving endpoint, not in the agent code.
"""

import mlflow
import re
import logging

logger = logging.getLogger(__name__)

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}

INJECTION_PATTERNS = [
    r"ignore (?:all )?(?:previous |above )?instructions",
    r"you are now",
    r"pretend (?:you are|to be)",
    r"forget (?:all |everything )",
    r"disregard (?:all |previous )",
    r"override (?:your |the )?(?:system|instructions)",
    r"new (?:system )?prompt",
    r"act as (?:a |an )?(?:different|new)",
    r"jailbreak",
    r"do anything now",
]

TOXICITY_KEYWORDS = [
    "kill", "murder", "attack", "bomb", "weapon", "hack",
    "steal", "fraud", "illegal", "drug", "abuse",
]


class PreLLMGuardrails:
    """
    Pre-LLM guardrail pipeline.
    Configurable via config dict — each check can be enabled/disabled.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.enabled_checks = config.get("enabled_checks", [
            "input_length_min", "input_length_max", "pii", "injection", "toxicity", "intent",
        ])
        self.max_input_length = config.get("max_input_length", 4000)
        self.min_input_length = config.get("min_input_length", 10)
        self.domain_context_keywords = config.get("domain_context_keywords", [])
        self.allowed_intents = config.get("allowed_intents", [])
        self.intent_keywords = config.get("intent_keywords", {})

    @mlflow.trace(span_type="GUARDRAIL", name="pre_llm_check")
    def check(self, user_message: str, conversation_context: str = "") -> dict:
        """Run all enabled pre-LLM guardrail checks in order.

        Args:
            conversation_context: Combined text from prior turns in the session.
                Used by the intent check to understand context for follow-up
                messages that may lack Databricks keywords on their own.
        """
        self._conversation_context = conversation_context
        results = {"blocked": False, "checks": {}}

        for check_name in self.enabled_checks:
            check_fn = getattr(self, f"_check_{check_name}", None)
            if check_fn:
                result = check_fn(user_message)
                results["checks"][check_name] = result
                if result.get("blocked"):
                    results["blocked"] = True
                    results["message"] = result.get("message", "Request blocked.")
                    results["blocked_by"] = check_name
                    logger.warning(f"Pre-LLM '{check_name}' blocked: {result.get('reason', '')}")
                    break

        return results

    def _check_input_length_min(self, message: str) -> dict:
        if len(message.strip()) < self.min_input_length:
            return {
                "blocked": True,
                "reason": f"Input too short ({len(message.strip())} < {self.min_input_length})",
                "message": "Your question is too short. Please provide more detail so I can help you.",
            }
        return {"blocked": False}

    def _check_input_length_max(self, message: str) -> dict:
        if len(message) > self.max_input_length:
            return {
                "blocked": True,
                "reason": f"Input too long ({len(message)} > {self.max_input_length})",
                "message": f"Your message is too long. Please keep it under {self.max_input_length} characters.",
            }
        return {"blocked": False}

    def _check_pii(self, message: str) -> dict:
        for pii_type, pattern in PII_PATTERNS.items():
            if re.findall(pattern, message):
                return {
                    "blocked": True,
                    "reason": f"PII detected: {pii_type}",
                    "message": "Your message contains personal information. Please remove sensitive data and try again.",
                }
        return {"blocked": False}

    def _check_injection(self, message: str) -> dict:
        message_lower = message.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, message_lower):
                return {
                    "blocked": True,
                    "reason": f"Prompt injection detected: {pattern}",
                    "message": "Your request could not be processed. Please ask a genuine question.",
                }
        return {"blocked": False}

    def _check_toxicity(self, message: str) -> dict:
        message_lower = message.lower()
        found = [kw for kw in TOXICITY_KEYWORDS if kw in message_lower]

        if found:
            has_context = any(ctx in message_lower for ctx in self.domain_context_keywords)
            if not has_context:
                return {
                    "blocked": True,
                    "reason": f"Toxic content: {found}",
                    "message": "Your message contains content that doesn't align with our usage policies.",
                }

        return {"blocked": False}

    def _check_intent(self, message: str) -> dict:
        """Intent check using a deny-list approach instead of allow-list.

        Instead of listing every allowed keyword (brittle, blocks legitimate users),
        we only block messages that are clearly NOT natural language input:
          - Gibberish / random characters
          - Extremely short with no words

        The LLM itself handles off-topic questions gracefully via the system prompt
        ("I'm a Databricks Documentation Assistant..."). If someone asks about cooking,
        the LLM will naturally redirect — no need to block at the guardrail level.
        """
        text = message.strip()

        # Block gibberish: messages with very few real words relative to length
        words = re.findall(r'[a-zA-Z]{2,}', text)
        if len(text) > 20 and len(words) < 2:
            return {
                "blocked": True,
                "intent": "gibberish",
                "reason": f"Message appears to be gibberish ({len(words)} words in {len(text)} chars)",
                "message": "I couldn't understand your message. Please ask a question in plain language.",
            }

        # Block messages that are just numbers/symbols with no words
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if len(text) > 10 and alpha_ratio < 0.3:
            return {
                "blocked": True,
                "intent": "non_text",
                "reason": f"Message is mostly non-alphabetic ({alpha_ratio:.0%} alpha)",
                "message": "I couldn't understand your message. Please ask a question in plain language.",
            }

        return {"blocked": False, "intent": "accepted"}

"""
Token Usage & Cost Tracker for Databricks LLM endpoints.

Extracts token counts from API response metadata and calculates cost.
Token counts come from the API response (not tiktoken) — works with
both OpenAI-compatible and Databricks-native response formats.

Usage:
    tracker = TokenTracker()
    tracker.track(response)  # from WorkspaceClient or OpenAI client
    print(tracker)           # Token usage summary
    tracker.to_tags()        # MLflow trace tags dict
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Databricks FMAPI model pricing (approximate $/1K tokens)
# Source: Databricks pricing page + model endpoint configs
MODEL_COST_PER_1K_TOKENS = {
    # Databricks-hosted OSS models (pay-per-token)
    "databricks-gpt-oss-120b": {"input": 0.001, "output": 0.002},
    "databricks-gpt-oss-20b": {"input": 0.0005, "output": 0.001},
    "databricks-meta-llama-3-3-70b-instruct": {"input": 0.00065, "output": 0.00225},
    "databricks-meta-llama-3-1-405b-instruct": {"input": 0.005, "output": 0.015},
    "databricks-meta-llama-3-1-70b-instruct": {"input": 0.00065, "output": 0.00225},
    "databricks-meta-llama-4-maverick": {"input": 0.0005, "output": 0.0015},
    "databricks-claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "databricks-gemini-2-0-flash": {"input": 0.0001, "output": 0.0004},
    "databricks-gemini-3-flash": {"input": 0.0001, "output": 0.0004},
    "databricks-gte-large-en": {"input": 0.0001, "output": 0.0},
    "databricks-bge-large-en": {"input": 0.0001, "output": 0.0},
    # OpenAI models (for reference)
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic models (for reference)
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
}


class TokenTracker:
    """Track token usage and cost across LLM calls."""

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.requests: int = 0

    def track(self, response) -> "TokenTracker":
        """Extract token usage from an LLM response and accumulate.

        Supports:
          - Databricks SDK response (response.usage)
          - OpenAI client response (response.usage)
          - Raw dict response (response["usage"])
        """
        usage = self._extract_usage(response)
        if usage:
            prompt = usage.get("prompt_tokens", 0) or 0
            completion = usage.get("completion_tokens", 0) or 0
            total = usage.get("total_tokens", 0) or (prompt + completion)

            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.requests += 1

            cost = self._calculate_cost(prompt, completion)
            self.total_cost += cost

            logger.debug(f"Token usage: {prompt}+{completion}={total} tokens, ${cost:.6f}")

        return self

    def track_streaming(self, chunks: list) -> "TokenTracker":
        """Extract token usage from the final streaming chunk.

        The last chunk in a streaming response often contains usage info.
        """
        if not chunks:
            return self

        # Check last chunk for usage
        last = chunks[-1] if isinstance(chunks, list) else chunks
        usage = self._extract_usage(last)
        if usage:
            return self.track(last)

        # If no usage in last chunk, estimate from chunk count
        # Rough estimate: ~1.3 tokens per chunk (average for SSE streaming)
        estimated_completion = int(len(chunks) * 1.3)
        self.completion_tokens += estimated_completion
        self.total_tokens += estimated_completion
        self.requests += 1
        self.total_cost += self._calculate_cost(0, estimated_completion)
        return self

    def _extract_usage(self, response) -> Optional[dict]:
        """Extract usage dict from various response formats."""
        if response is None:
            return None

        # Object with .usage attribute (SDK response, OpenAI response)
        if hasattr(response, "usage") and response.usage is not None:
            usage_obj = response.usage
            if isinstance(usage_obj, dict):
                return usage_obj
            return {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

        # Dict response
        if isinstance(response, dict):
            if "usage" in response:
                return response["usage"]

        return None

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD based on model pricing."""
        model = self.model_name.lower()

        # Try exact match first, then prefix match
        pricing = MODEL_COST_PER_1K_TOKENS.get(model)
        if not pricing:
            for name, p in MODEL_COST_PER_1K_TOKENS.items():
                if name in model or model in name:
                    pricing = p
                    break

        if not pricing:
            return 0.0

        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return input_cost + output_cost

    def to_tags(self) -> dict:
        """Return MLflow trace tags for token usage."""
        return {
            "agentops.tokens.prompt": str(self.prompt_tokens),
            "agentops.tokens.completion": str(self.completion_tokens),
            "agentops.tokens.total": str(self.total_tokens),
            "agentops.cost_usd": f"{self.total_cost:.6f}",
            "agentops.llm_requests": str(self.requests),
        }

    def __repr__(self) -> str:
        return (
            f"Tokens: {self.prompt_tokens} prompt + {self.completion_tokens} completion "
            f"= {self.total_tokens} total | Cost: ${self.total_cost:.6f} | "
            f"Requests: {self.requests}"
        )

"""
Token Usage & Cost Tracker for Databricks LLM endpoints.

Extracts token counts from API response metadata and calculates cost.
Token counts come from the API response (not tiktoken) — works with
both OpenAI-compatible and Databricks-native response formats.

Supports cached token pricing — when prompt caching is enabled,
cached input tokens are charged at a reduced rate.

Usage:
    tracker = TokenTracker()
    tracker.track(response)  # from WorkspaceClient or OpenAI client
    print(tracker)           # Token usage summary
    tracker.to_tags()        # MLflow trace tags dict
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Databricks Foundation Model API pricing ($/1M tokens) ──────
# Converted to $/1K tokens for calculation.
# Source: https://www.databricks.com/product/pricing/foundation-model-serving
# Format: {"input": $/1K, "output": $/1K, "cached_input": $/1K}
# cached_input is the discounted rate when prompt caching is used.

MODEL_COST_PER_1K_TOKENS = {
    # ── OpenAI (via Databricks FMAPI) ──
    "databricks-gpt-5-5-pro":      {"input": 0.010,   "output": 0.030,   "cached_input": 0.005},
    "databricks-gpt-5-5":          {"input": 0.005,   "output": 0.020,   "cached_input": 0.0025},
    "databricks-gpt-5-4":          {"input": 0.0025,  "output": 0.010,   "cached_input": 0.00125},
    "databricks-gpt-5-4-mini":     {"input": 0.0004,  "output": 0.0016,  "cached_input": 0.0002},
    "databricks-gpt-5-4-nano":     {"input": 0.0002,  "output": 0.0008,  "cached_input": 0.0001},
    "databricks-gpt-5-3-codex":    {"input": 0.003,   "output": 0.012,   "cached_input": 0.0015},
    "databricks-gpt-5-2-codex":    {"input": 0.003,   "output": 0.012,   "cached_input": 0.0015},
    "databricks-gpt-5-2":          {"input": 0.0025,  "output": 0.010,   "cached_input": 0.00125},
    "databricks-gpt-5-1":          {"input": 0.002,   "output": 0.008,   "cached_input": 0.001},
    "databricks-gpt-5-1-codex-max": {"input": 0.003,  "output": 0.012,   "cached_input": 0.0015},
    "databricks-gpt-5-1-codex-mini": {"input": 0.00075, "output": 0.003, "cached_input": 0.000375},
    "databricks-gpt-5":            {"input": 0.002,   "output": 0.008,   "cached_input": 0.001},
    "databricks-gpt-5-mini":       {"input": 0.0003,  "output": 0.0012,  "cached_input": 0.00015},
    "databricks-gpt-5-nano":       {"input": 0.00015, "output": 0.0006,  "cached_input": 0.000075},
    "databricks-gpt-oss-120b":     {"input": 0.001,   "output": 0.002,   "cached_input": 0.0005},
    "databricks-gpt-oss-20b":      {"input": 0.0005,  "output": 0.001,   "cached_input": 0.00025},

    # ── Anthropic Claude (via Databricks FMAPI) ──
    "databricks-claude-opus-4-7":   {"input": 0.015,  "output": 0.075,   "cached_input": 0.0075},
    "databricks-claude-opus-4-6":   {"input": 0.015,  "output": 0.075,   "cached_input": 0.0075},
    "databricks-claude-opus-4-5":   {"input": 0.015,  "output": 0.075,   "cached_input": 0.0075},
    "databricks-claude-opus-4-1":   {"input": 0.015,  "output": 0.075,   "cached_input": 0.0075},
    "databricks-claude-sonnet-4-6": {"input": 0.003,  "output": 0.015,   "cached_input": 0.0015},
    "databricks-claude-sonnet-4-5": {"input": 0.003,  "output": 0.015,   "cached_input": 0.0015},
    "databricks-claude-sonnet-4":   {"input": 0.003,  "output": 0.015,   "cached_input": 0.0015},
    "databricks-claude-haiku-4-5":  {"input": 0.0008, "output": 0.004,   "cached_input": 0.0004},

    # ── Google Gemini (via Databricks FMAPI) ──
    "databricks-gemini-3-1-flash-lite": {"input": 0.0001, "output": 0.0004, "cached_input": 0.00005},
    "databricks-gemini-3-flash":    {"input": 0.0001,  "output": 0.0004,  "cached_input": 0.00005},
    "databricks-gemini-3-1-pro":    {"input": 0.00125, "output": 0.005,   "cached_input": 0.000625},
    "databricks-gemini-3-pro":      {"input": 0.00125, "output": 0.005,   "cached_input": 0.000625},
    "databricks-gemini-2-5-pro":    {"input": 0.00125, "output": 0.01,    "cached_input": 0.000625},
    "databricks-gemini-2-5-flash":  {"input": 0.00015, "output": 0.0006,  "cached_input": 0.000075},
    "databricks-gemma-3-12b":       {"input": 0.0001,  "output": 0.0003,  "cached_input": 0.00005},

    # ── Meta Llama (via Databricks FMAPI) ──
    "databricks-llama-4-maverick":  {"input": 0.0005,  "output": 0.0015,  "cached_input": 0.00025},
    "databricks-meta-llama-3-3-70b-instruct":  {"input": 0.00065, "output": 0.00225, "cached_input": 0.000325},
    "databricks-meta-llama-3-1-405b-instruct": {"input": 0.005,   "output": 0.015,   "cached_input": 0.0025},
    "databricks-meta-llama-3-1-70b-instruct":  {"input": 0.00065, "output": 0.00225, "cached_input": 0.000325},
    "databricks-meta-llama-3-1-8b-instruct":   {"input": 0.00015, "output": 0.0006,  "cached_input": 0.000075},

    # ── Alibaba Qwen (via Databricks FMAPI) ──
    "databricks-qwen35-122b-a10b":  {"input": 0.0003, "output": 0.0012,  "cached_input": 0.00015},
    "databricks-qwen3-next-80b-a3b-instruct": {"input": 0.0005, "output": 0.0015, "cached_input": 0.00025},

    # ── Embedding models ──
    "databricks-gte-large-en":      {"input": 0.0001, "output": 0.0,     "cached_input": 0.0001},
    "databricks-bge-large-en":      {"input": 0.0001, "output": 0.0,     "cached_input": 0.0001},
    "databricks-qwen3-embedding-0-6b": {"input": 0.0001, "output": 0.0,  "cached_input": 0.0001},

    # ── OpenAI direct (for reference/comparison) ──
    "gpt-4o":                       {"input": 0.0025, "output": 0.010,   "cached_input": 0.00125},
    "gpt-4o-mini":                  {"input": 0.00015, "output": 0.0006, "cached_input": 0.000075},
    "gpt-4-turbo":                  {"input": 0.010,  "output": 0.030,   "cached_input": 0.005},
    "gpt-3.5-turbo":                {"input": 0.0005, "output": 0.0015,  "cached_input": 0.00025},

    # ── Anthropic direct (for reference/comparison) ──
    "claude-sonnet-4-20250514":     {"input": 0.003,  "output": 0.015,   "cached_input": 0.0015},
    "claude-3-5-sonnet-20241022":   {"input": 0.003,  "output": 0.015,   "cached_input": 0.0015},
    "claude-3-haiku-20240307":      {"input": 0.00025, "output": 0.00125, "cached_input": 0.000125},
    "claude-opus-4-20250514":       {"input": 0.015,  "output": 0.075,   "cached_input": 0.0075},
}


class TokenTracker:
    """Track token usage and cost across LLM calls.

    Supports cached token pricing — when the API response includes
    cached_tokens in the usage metadata, those tokens are charged
    at the reduced cached_input rate.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cached_tokens: int = 0
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
            cached = usage.get("cached_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)) or 0
            total = usage.get("total_tokens", 0) or (prompt + completion)

            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cached_tokens += cached
            self.total_tokens += total
            self.requests += 1

            cost = self._calculate_cost(prompt, completion, cached)
            self.total_cost += cost

            logger.debug(f"Token usage: {prompt}+{completion}={total} tokens "
                         f"(cached: {cached}), ${cost:.6f}")

        return self

    def track_streaming(self, chunks: list) -> "TokenTracker":
        """Extract token usage from the final streaming chunk.

        The last chunk in a streaming response often contains usage info.
        """
        if not chunks:
            return self

        last = chunks[-1] if isinstance(chunks, list) else chunks
        usage = self._extract_usage(last)
        if usage:
            return self.track(last)

        # If no usage in last chunk, estimate from chunk count
        estimated_completion = int(len(chunks) * 1.3)
        self.completion_tokens += estimated_completion
        self.total_tokens += estimated_completion
        self.requests += 1
        self.total_cost += self._calculate_cost(0, estimated_completion, 0)
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
            result = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }
            # Extract cached tokens from prompt_tokens_details if available
            details = getattr(usage_obj, "prompt_tokens_details", None)
            if details:
                if isinstance(details, dict):
                    result["cached_tokens"] = details.get("cached_tokens", 0)
                else:
                    result["cached_tokens"] = getattr(details, "cached_tokens", 0)
            return result

        # Dict response
        if isinstance(response, dict) and "usage" in response:
            return response["usage"]

        return None

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int,
                        cached_tokens: int = 0) -> float:
        """Calculate cost in USD based on model pricing.

        When cached_tokens > 0, those tokens are charged at the
        cached_input rate instead of the full input rate.
        """
        model = self.model_name.lower()

        # Try exact match first, then prefix/substring match
        pricing = MODEL_COST_PER_1K_TOKENS.get(model)
        if not pricing:
            for name, p in MODEL_COST_PER_1K_TOKENS.items():
                if name in model or model in name:
                    pricing = p
                    break

        if not pricing:
            return 0.0

        # Split input tokens into cached and non-cached
        non_cached_input = max(0, prompt_tokens - cached_tokens)
        cached_rate = pricing.get("cached_input", pricing["input"] * 0.5)

        input_cost = (non_cached_input / 1000) * pricing["input"]
        cached_cost = (cached_tokens / 1000) * cached_rate
        output_cost = (completion_tokens / 1000) * pricing["output"]

        return input_cost + cached_cost + output_cost

    def to_tags(self) -> dict:
        """Return MLflow trace tags for token usage."""
        tags = {
            "agentops.tokens.prompt": str(self.prompt_tokens),
            "agentops.tokens.completion": str(self.completion_tokens),
            "agentops.tokens.total": str(self.total_tokens),
            "agentops.cost_usd": f"{self.total_cost:.6f}",
            "agentops.llm_requests": str(self.requests),
        }
        if self.cached_tokens > 0:
            tags["agentops.tokens.cached"] = str(self.cached_tokens)
        return tags

    def to_dict(self) -> dict:
        """Return full usage as a dict (for logging, audit)."""
        return {
            "model": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "requests": self.requests,
        }

    def __repr__(self) -> str:
        cached_str = f" (cached: {self.cached_tokens})" if self.cached_tokens > 0 else ""
        return (
            f"Tokens: {self.prompt_tokens} prompt + {self.completion_tokens} completion "
            f"= {self.total_tokens} total{cached_str} | Cost: ${self.total_cost:.6f} | "
            f"Requests: {self.requests}"
        )

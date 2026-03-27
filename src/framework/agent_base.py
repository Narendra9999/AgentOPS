"""
AgentOPS Framework — Base Agent Class
All agents must extend this. Provides standardized guardrails,
MLflow tracing with tags, and audit logging.
"""

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from typing import Optional
import logging
import uuid
import time

from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.guardrails.post_llm import PostLLMGuardrails

logger = logging.getLogger(__name__)


class AgentOPSBase(ChatAgent):
    """
    Base agent class for the AgentOPS framework.
    Wraps agent logic with pre/post LLM guardrails and MLflow tracing.
    Subclasses must implement _process_request().
    """

    def __init__(self, config: dict):
        self.config = config
        agent_cfg = config.get("agent", {})
        self.agent_name = agent_cfg.get("name", "unnamed_agent")
        self.agent_version = agent_cfg.get("version", "0.0.0")

        # Initialize guardrails from config
        gr_config = config.get("guardrails", {})
        self.guardrails_enabled = gr_config.get("enabled", True)
        self.pre_llm_guardrails = PreLLMGuardrails(gr_config.get("pre_llm", {}))
        self.post_llm_guardrails = PostLLMGuardrails(gr_config.get("post_llm", {}))

        # Tracing tags — set on every trace automatically
        # Environment is resolved at runtime, not from config.yaml
        tracing_config = config.get("tracing", {})
        environment = self._resolve_environment(tracing_config)
        self.default_tags = {
            "agentops.agent_name": self.agent_name,
            "agentops.agent_version": self.agent_version,
            "agentops.framework_version": "0.1.0",
            "agentops.environment": environment,
            **tracing_config.get("default_tags", {}),
        }

    @staticmethod
    def _resolve_environment(tracing_config: dict) -> str:
        """
        Resolve environment dynamically. Priority:
        1. AGENTOPS_ENVIRONMENT env var (set by DABs/Jenkins)
        2. tracing.default_tags.agentops.environment in config.yaml
        3. Detect from workspace URL
        4. Default: "unknown"
        """
        import os

        # 1. Env var (set by CI/CD or DABs target)
        env = os.environ.get("AGENTOPS_ENVIRONMENT")
        if env:
            return env

        # 2. Config.yaml value
        config_env = tracing_config.get("default_tags", {}).get("agentops.environment")
        if config_env:
            return config_env

        # 3. Try to detect from workspace
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            host = w.config.host or ""
            if "stage" in host.lower() or "staging" in host.lower():
                return "stage"
            elif "prod" in host.lower():
                return "prod"
            else:
                return "dev"
        except Exception:
            pass

        return "unknown"

    def _set_trace_tags(self, extra_tags: Optional[dict] = None):
        tags = {**self.default_tags}
        if extra_tags:
            tags.update(extra_tags)
        for key, value in tags.items():
            mlflow.update_current_trace(tags={key: str(value)})

    @mlflow.trace(span_type="AGENT")
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ) -> ChatAgentResponse:
        """
        Main prediction method with guardrail wrapping.
        Subclasses should NOT override this — override _process_request() instead.
        """
        start_time = time.time()
        user_message = messages[-1].content if messages else ""
        self._request_context = {}

        # Extract per-request tags
        request_tags = {}
        if custom_inputs and isinstance(custom_inputs, dict):
            request_tags = custom_inputs.get("tags", {})
        self._set_trace_tags(request_tags)

        # ── Pre-LLM Guardrails ──
        if self.guardrails_enabled:
            pre_result = self.pre_llm_guardrails.check(user_message)
            if pre_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.pre_llm.blocked": "true",
                    "agentops.guardrail.pre_llm.blocked_by": pre_result.get("blocked_by", "unknown"),
                })
                return ChatAgentResponse(
                    messages=[ChatAgentMessage(
                        id=str(uuid.uuid4()), role="assistant",
                        content=pre_result.get("message", "Request blocked by safety filters."),
                    )]
                )
            mlflow.update_current_trace(tags={
                "agentops.guardrail.pre_llm.blocked": "false",
                "agentops.guardrail.pre_llm.intent": pre_result.get("checks", {}).get("intent", {}).get("intent", "unknown"),
            })

        # ── Agent Processing (subclass implements this) ──
        # _request_context is a plain dict that subclasses can populate
        # with data for post-LLM guardrails (e.g., retrieved_docs).
        # This avoids mutating the MLflow ChatContext pydantic object.
        self._request_context = {}
        response = self._process_request(messages, context, custom_inputs)

        # ── Post-LLM Guardrails ──
        if self.guardrails_enabled:
            response_text = response.messages[-1].content if response.messages else ""
            post_result = self.post_llm_guardrails.check(user_message, response_text, self._request_context)
            if post_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.post_llm.blocked": "true",
                    "agentops.guardrail.post_llm.blocked_by": post_result.get("blocked_by", "unknown"),
                })
                return ChatAgentResponse(
                    messages=[ChatAgentMessage(
                        id=str(uuid.uuid4()), role="assistant",
                        content=post_result.get("message", "Response filtered for safety."),
                    )]
                )
            mlflow.update_current_trace(tags={"agentops.guardrail.post_llm.blocked": "false"})

        # ── Tag with latency ──
        latency_ms = (time.time() - start_time) * 1000
        mlflow.update_current_trace(tags={
            "agentops.latency_ms": str(round(latency_ms, 2)),
            "agentops.status": "success",
        })

        return response

    def _process_request(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ) -> ChatAgentResponse:
        """Override this in your agent subclass."""
        raise NotImplementedError("Subclasses must implement _process_request")

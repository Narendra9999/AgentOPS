"""
AgentOPS Framework — Base Agent Class
All agents must extend this. Provides standardized guardrails,
MLflow tracing with tags, session history, and long-term user memory.
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
from framework.session.session_store import SessionStore

logger = logging.getLogger(__name__)


class AgentOPSBase(ChatAgent):
    """
    Base agent class for the AgentOPS framework.
    Wraps agent logic with pre/post LLM guardrails and MLflow tracing.
    Subclasses must implement _process_request().

    Memory support (via Lakebase DatabricksStore):
      - Short-term: per-thread conversation history (auto-loaded/saved)
      - Long-term: cross-session user memory recalled into _request_context["user_memories"]

    Clients pass thread_id and user_id via custom_inputs:
      {"thread_id": "abc-123", "user_id": "user@example.com"}
    Response includes custom_outputs with thread_id for follow-up requests.
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

        # Session history + long-term memory
        self.session_store = SessionStore(config)

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

        # Extract per-request tags, thread_id, and user_id
        request_tags = {}
        thread_id = ""
        user_id = ""
        if custom_inputs and isinstance(custom_inputs, dict):
            request_tags = custom_inputs.get("tags", {})
            # Support both thread_id (new) and session_id (legacy) naming
            thread_id = custom_inputs.get("thread_id", custom_inputs.get("session_id", ""))
            user_id = custom_inputs.get("user_id", "")
        if not thread_id:
            try:
                trace = mlflow.get_current_active_span()
                thread_id = trace.request_id if trace else ""
            except Exception:
                pass
        if not thread_id:
            thread_id = str(uuid.uuid4())
        self._set_trace_tags(request_tags)

        # ── Load session history FIRST (before guardrails) ──
        # History must load before guardrails so the intent check can see
        # conversation context from prior turns. Without this, follow-up
        # messages like "What happens if I set it too low?" get blocked
        # because they lack Databricks keywords on their own.
        has_client_history = len(messages) > 1
        conversation_context = ""
        if self.session_store.enabled and thread_id and not has_client_history:
            self._load_session_history(thread_id, messages)
            messages = self._augmented_messages
            # Build context string from prior user messages for intent detection
            if len(messages) > 1:
                conversation_context = " ".join(
                    m.content for m in messages[:-1] if hasattr(m, "role") and m.role == "user"
                )
        elif has_client_history:
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": str(len(messages) - 1),
                "agentops.session.source": "client",
            })

        # ── Pre-LLM Guardrails ──
        # Runs AFTER session history load so the intent check can see
        # conversation context from prior turns via conversation_context.
        if self.guardrails_enabled:
            pre_result = self.pre_llm_guardrails.check(
                user_message, conversation_context=conversation_context)
            if pre_result.get("blocked"):
                mlflow.update_current_trace(tags={
                    "agentops.guardrail.pre_llm.blocked": "true",
                    "agentops.guardrail.pre_llm.blocked_by": pre_result.get("blocked_by", "unknown"),
                })
                return self._build_response(
                    pre_result.get("message", "Request blocked by safety filters."),
                    thread_id, user_id,
                )
            mlflow.update_current_trace(tags={
                "agentops.guardrail.pre_llm.blocked": "false",
                "agentops.guardrail.pre_llm.intent": pre_result.get("checks", {}).get("intent", {}).get("intent", "unknown"),
            })

        # ── Recall long-term user memories ──
        # If user_id is provided, search Lakebase for relevant facts about this user
        # and make them available to the agent via _request_context.
        if self.session_store.memory_enabled and user_id and user_message:
            self._recall_long_term_memory(user_id, user_message)

        # ── Agent Processing (subclass implements this) ──
        # _request_context is a plain dict that subclasses can populate
        # with data for post-LLM guardrails (e.g., retrieved_docs).
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
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
                return self._build_response(
                    post_result.get("message", "Response filtered for safety."),
                    thread_id, user_id,
                )
            mlflow.update_current_trace(tags={"agentops.guardrail.post_llm.blocked": "false"})

        # ── Tag with latency ──
        latency_ms = (time.time() - start_time) * 1000
        mlflow.update_current_trace(tags={
            "agentops.latency_ms": str(round(latency_ms, 2)),
            "agentops.status": "success",
        })

        # ── Save session history ──
        if self.session_store.enabled:
            response_text = response.messages[-1].content if response.messages else ""
            trace_id = ""
            try:
                trace = mlflow.get_current_active_span()
                trace_id = trace.request_id if trace else ""
            except Exception:
                pass

            model_endpoint = getattr(self, "llm_endpoint", "")
            self.session_store.save_full_session(
                session_id=thread_id,
                messages=messages,
                response_text=response_text,
                response_time_ms=latency_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
            )

        # ── Add custom_outputs (thread_id for follow-up requests) ──
        custom_outputs = {"thread_id": thread_id}
        if user_id:
            custom_outputs["user_id"] = user_id
        # Diagnostic: surface store status for debugging (prefix with _ to avoid conflicts)
        store_err = getattr(self.session_store, '_store_error', None)
        if store_err:
            custom_outputs["_store_error"] = store_err[:200]
        uc_err = getattr(self.session_store, '_uc_last_error', None)
        if uc_err:
            custom_outputs["_uc_error"] = uc_err[:200]
        if response.custom_outputs:
            custom_outputs.update(response.custom_outputs)
        response.custom_outputs = custom_outputs

        return response

    @mlflow.trace(name="load_session_history", span_type="RETRIEVER")
    def _load_session_history(
        self, thread_id: str, messages: list[ChatAgentMessage]
    ) -> dict:
        """Load conversation history from Lakebase and prepend to messages.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so the loaded history is clearly visible in the MLflow trace output.
        """
        # Surface store connection status
        store_ok = self.session_store._store is not None
        store_err = getattr(self.session_store, '_store_error', None)
        mlflow.update_current_trace(tags={
            "agentops.session.store_connected": str(store_ok),
        })
        if store_err:
            mlflow.update_current_trace(tags={
                "agentops.session.store_error": store_err[:200],
            })

        prior_turns = self.session_store.get_history(thread_id, max_turns=10)
        if prior_turns:
            history_msgs = [
                ChatAgentMessage(role=t["role"], content=t["content"])
                for t in prior_turns
            ]
            self._augmented_messages = history_msgs + list(messages)
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": str(len(prior_turns)),
                "agentops.session.source": "server_store",
            })
            return {
                "thread_id": thread_id,
                "store_connected": store_ok,
                "turns_loaded": len(prior_turns),
                "messages_loaded": [{"role": t["role"], "content": t["content"][:100]} for t in prior_turns],
            }
        else:
            self._augmented_messages = list(messages)
            mlflow.update_current_trace(tags={
                "agentops.session.history_turns": "0",
                "agentops.session.source": "server_store",
            })
            return {
                "thread_id": thread_id,
                "store_connected": store_ok,
                "store_error": store_err,
                "turns_loaded": 0,
            }

    @mlflow.trace(name="recall_user_memory", span_type="RETRIEVER")
    def _recall_long_term_memory(self, user_id: str, query: str) -> dict:
        """Search long-term user memory in Lakebase.

        Traced as a RETRIEVER span — returns a dict with diagnostic info
        so recalled memories are clearly visible in the MLflow trace output.
        """
        memories = self.session_store.recall_user_memories(user_id, query)
        if memories:
            self._request_context["user_memories"] = memories
            mlflow.update_current_trace(tags={
                "agentops.memory.recalled_count": str(len(memories)),
            })
        return {
            "user_id": user_id,
            "query": query[:100],
            "memories_found": len(memories),
            "memories": memories,
        }

    def _build_response(
        self, content: str, thread_id: str, user_id: str = ""
    ) -> ChatAgentResponse:
        """Build a ChatAgentResponse with standard custom_outputs."""
        custom_outputs = {"thread_id": thread_id}
        if user_id:
            custom_outputs["user_id"] = user_id
        return ChatAgentResponse(
            messages=[ChatAgentMessage(
                id=str(uuid.uuid4()), role="assistant", content=content,
            )],
            custom_outputs=custom_outputs,
        )

    def _process_request(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ) -> ChatAgentResponse:
        """Override this in your agent subclass."""
        raise NotImplementedError("Subclasses must implement _process_request")

    def _process_request_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ):
        """Override this for streaming support. Yields ChatAgentChunk objects."""
        raise NotImplementedError("Subclasses must implement _process_request_stream for streaming")

    @mlflow.trace(span_type="AGENT")
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ):
        """
        Streaming prediction — same pre/post processing as predict(),
        but yields chunks as the LLM generates tokens.

        Pre-processing (history, guardrails, memory) runs fully before streaming.
        Post-processing (guardrails, save history) runs after streaming completes.
        """
        start_time = time.time()
        user_message = messages[-1].content if messages else ""
        self._request_context = {}

        # Extract thread_id, user_id
        request_tags = {}
        thread_id = ""
        user_id = ""
        if custom_inputs and isinstance(custom_inputs, dict):
            request_tags = custom_inputs.get("tags", {})
            thread_id = custom_inputs.get("thread_id", custom_inputs.get("session_id", ""))
            user_id = custom_inputs.get("user_id", "")
        if not thread_id:
            thread_id = str(uuid.uuid4())
        self._set_trace_tags(request_tags)

        # ── Load session history ──
        has_client_history = len(messages) > 1
        conversation_context = ""
        if self.session_store.enabled and thread_id and not has_client_history:
            self._load_session_history(thread_id, messages)
            messages = self._augmented_messages
            if len(messages) > 1:
                conversation_context = " ".join(
                    m.content for m in messages[:-1] if hasattr(m, "role") and m.role == "user"
                )

        # ── Pre-LLM Guardrails ──
        if self.guardrails_enabled:
            pre_result = self.pre_llm_guardrails.check(
                user_message, conversation_context=conversation_context)
            if pre_result.get("blocked"):
                from mlflow.types.agent import ChatAgentChunk, ChatAgentChunkChoice, ChatAgentChunkChoiceDelta
                yield ChatAgentChunk(
                    choices=[ChatAgentChunkChoice(
                        delta=ChatAgentChunkChoiceDelta(
                            role="assistant",
                            content=pre_result.get("message", "Request blocked."),
                        )
                    )]
                )
                return

        # ── Recall long-term memory ──
        if self.session_store.memory_enabled and user_id and user_message:
            self._recall_long_term_memory(user_id, user_message)

        # ── Stream agent response ──
        self._request_context["thread_id"] = thread_id
        self._request_context["user_id"] = user_id
        full_response = []

        for chunk in self._process_request_stream(messages, context, custom_inputs):
            full_response.append(chunk.choices[0].delta.content if chunk.choices else "")
            yield chunk

        response_text = "".join(full_response)

        # ── Post-LLM Guardrails ──
        if self.guardrails_enabled:
            post_result = self.post_llm_guardrails.check(
                user_message, response_text, self._request_context)
            if post_result.get("blocked"):
                logger.warning(f"Post-LLM guardrail blocked streamed response: {post_result.get('blocked_by')}")
                # Can't un-stream — log the block for monitoring

        # ── Save session history ──
        latency_ms = (time.time() - start_time) * 1000
        if self.session_store.enabled:
            trace_id = ""
            try:
                trace = mlflow.get_current_active_span()
                trace_id = trace.request_id if trace else ""
            except Exception:
                pass
            model_endpoint = getattr(self, "llm_endpoint", "")
            self.session_store.save_full_session(
                session_id=thread_id,
                messages=messages,
                response_text=response_text,
                response_time_ms=latency_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
            )

        mlflow.update_current_trace(tags={
            "agentops.latency_ms": str(round(latency_ms, 2)),
            "agentops.status": "success",
            "agentops.streaming": "true",
        })

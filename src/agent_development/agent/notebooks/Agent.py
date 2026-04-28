"""
Databricks Documentation Assistant — Agent Implementation
Extends AgentOPSBase with RAG-based documentation retrieval.

Dependencies (bundled with the model):
  - framework/ (via code_paths) — AgentOPSBase, guardrails, tracing
  - tools/ (via code_paths) — search_docs for vector search
  - config.yaml (via model_config) — all configuration
  - databricks-sdk (via pip_requirements) — SDK for VS and LLM calls
"""

import mlflow
from mlflow.models import ModelConfig
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse
from typing import Optional
import uuid
import logging

from framework.agent_base import AgentOPSBase
from tools.agent_tools import search_docs

logger = logging.getLogger(__name__)


class DatabricksDocsAgent(AgentOPSBase):
    """
    RAG agent for Databricks documentation.

    Inherits from AgentOPSBase which provides:
      - Pre-LLM guardrails (PII, injection, toxicity, intent)
      - Post-LLM guardrails (compliance, PII leakage, hallucination, quality)
      - MLflow tracing with tags
      - Dynamic environment detection

    AI Gateway safety filter provides additional LLM-based content safety
    at the serving endpoint level.
    """

    def __init__(self):
        config = ModelConfig(development_config="config.yaml").to_dict()
        super().__init__(config)

        # All values from config.yaml — no hardcoding
        self.llm_endpoint = config["llm"]["endpoint"]
        self.max_tokens = config["llm"].get("max_tokens", 2048)
        self.temperature = config["llm"].get("temperature", 0.1)
        self.max_history_turns = config["llm"].get("max_history_turns", 10)
        self.max_retrieval_turns = config["llm"].get("max_retrieval_turns", 3)
        # Load system prompt: MLflow Prompt Registry → config.yaml fallback
        self.system_prompt = self._load_system_prompt(config)

        # Vector search — fully qualified index name from config
        vs_config = config.get("vector_search", {})
        catalog = config.get("catalog", "")
        schema = config.get("schema", "")
        index_name = vs_config.get("index", "")

        # Build fully qualified index name if not already qualified
        if index_name and "." not in index_name and catalog and schema:
            self.vs_index = f"{catalog}.{schema}.{index_name}"
        else:
            self.vs_index = index_name

        self.vs_num_results = vs_config.get("num_results", 5)
        self.vs_columns = vs_config.get("columns", ["chunk_text", "url", "chunk_id"])

    def _load_system_prompt(self, config: dict) -> str:
        """Load system prompt from MLflow Prompt Registry, fall back to config.yaml.

        The prompt is registered during the pipeline's RegisterModel step and
        can be versioned, A/B tested, and updated without redeploying the model.
        """
        catalog = config.get("catalog", "")
        schema = config.get("schema", "")
        agent_name = config.get("agent", {}).get("name", "")
        prompt_name = f"{catalog}.{schema}.{agent_name}_system_prompt"

        try:
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@production")
            logger.info(f"Loaded prompt from registry: {prompt_name}@production (v{prompt.version})")
            return prompt.template
        except Exception as e:
            logger.info(f"Prompt registry unavailable ({e}), using config.yaml")
            return config.get("system_prompt", "")

    @mlflow.trace(span_type="RETRIEVER")
    def _retrieve_context(self, query: str) -> tuple[str, list[dict]]:
        """Retrieve relevant documentation chunks via vector search."""
        docs = search_docs(
            query=query,
            index_name=self.vs_index,
            columns=self.vs_columns,
            num_results=self.vs_num_results,
        )

        if not docs:
            return "No relevant documentation found.", []

        context_parts = []
        for i, doc in enumerate(docs, 1):
            url = doc.get("url", "")
            content = doc.get(self.vs_columns[0], doc.get("content", ""))
            context_parts.append(f"[Source {i}] {url}\n{content}")

        return "\n\n---\n\n".join(context_parts), docs

    @mlflow.trace(span_type="LLM")
    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM endpoint."""
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        w = WorkspaceClient()
        chat_messages = [
            ChatMessage(
                role=ChatMessageRole.SYSTEM if m["role"] == "system"
                     else ChatMessageRole.USER if m["role"] == "user"
                     else ChatMessageRole.ASSISTANT,
                content=m["content"]
            ) for m in messages
        ]
        response = w.serving_endpoints.query(
            name=self.llm_endpoint,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        # Some models return content as a list of objects (reasoning + text)
        if isinstance(content, list):
            text_parts = [
                item.get("text", item.get("content", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") in ("text", "output_text", None)
            ]
            content = "\n".join(text_parts) if text_parts else str(content)
        return content

    @mlflow.trace(span_type="CHAIN")
    def _process_request(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ) -> ChatAgentResponse:
        # Build retrieval query from recent user turns
        recent_user_msgs = [
            m.content for m in messages if m.role == "user"
        ][-self.max_retrieval_turns:]
        retrieval_query = " ".join(recent_user_msgs)

        # Retrieve documentation context
        context_text, retrieved_docs = self._retrieve_context(retrieval_query)

        # Build augmented prompt
        augmented_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Documentation Context:\n\n{context_text}"},
        ]

        # Inject long-term user memories (recalled by AgentOPSBase from Lakebase)
        user_memories = self._request_context.get("user_memories", [])
        if user_memories:
            memory_lines = [f"  [{m['key']}]: {m['content']}" for m in user_memories]
            memory_text = "User context from prior sessions:\n" + "\n".join(memory_lines)
            augmented_messages.append({"role": "system", "content": memory_text})
        history = messages[-self.max_history_turns:] if len(messages) > self.max_history_turns else messages
        for msg in history:
            augmented_messages.append({"role": msg.role, "content": msg.content})

        # Call LLM
        response_text = self._call_llm(augmented_messages)

        # Store retrieved docs for post-LLM hallucination check.
        # Uses _request_context (plain dict set by AgentOPSBase.predict)
        # instead of the context param which may be a pydantic ChatContext.
        self._request_context["retrieved_docs"] = retrieved_docs

        return ChatAgentResponse(
            messages=[ChatAgentMessage(
                id=str(uuid.uuid4()), role="assistant", content=response_text)])


# MLflow ChatAgent entry point
AGENT = DatabricksDocsAgent()
mlflow.models.set_model(AGENT)

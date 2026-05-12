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
import re
import logging

from framework.agent_base import AgentOPSBase
from tools.agent_tools import search_docs, calculate, get_current_timestamp, format_sql, cluster_sizing, get_node_info, NODE_CATALOG
from tools.tool_loader import load_custom_tools, execute_custom_tools
from tools.token_tracker import TokenTracker

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
        self.vs_reranker_enabled = str(vs_config.get("reranker_enabled", "false")).lower() == "true"
        self.vs_reranker_model = vs_config.get("reranker_model", "")

        # Register shared tools
        self.tools = {
            "search_docs": search_docs,
            "calculate": calculate,
            "get_current_timestamp": get_current_timestamp,
            "format_sql": format_sql,
            "cluster_sizing": cluster_sizing,
            "get_node_info": get_node_info,
        }

        # Load team-specific custom tools from config
        tools_config = config.get("tools", {})
        custom_tool_names = tools_config.get("custom_tools", [])
        self.custom_tools = load_custom_tools(custom_tool_names)
        if self.custom_tools:
            logger.info(f"Loaded {len(self.custom_tools)} custom tools: {list(self.custom_tools.keys())}")

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
            reranker_enabled=self.vs_reranker_enabled,
            reranker_model=self.vs_reranker_model,
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
        """Call the LLM endpoint (non-streaming)."""
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

        # Track token usage
        tracker = TokenTracker(model_name=self.llm_endpoint)
        tracker.track(response)
        self._request_context["token_usage"] = tracker

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

    def _call_llm_stream(self, messages: list[dict]):
        """Stream LLM response token by token. Yields content chunks.

        Creates WorkspaceClient per-request (required for serving OBO auth),
        uses the OpenAI-compatible client for streaming.
        """
        from databricks.sdk import WorkspaceClient

        # Per-request client — required for model serving OBO credentials
        w = WorkspaceClient()
        openai_client = w.serving_endpoints.get_open_ai_client()

        tracker = TokenTracker(model_name=self.llm_endpoint)
        chunk_count = 0
        last_chunk_with_usage = None

        for chunk in openai_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        ):
            # Capture the last chunk with usage (only track once at the end)
            if hasattr(chunk, "usage") and chunk.usage:
                last_chunk_with_usage = chunk

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Some models stream content as a list of objects (reasoning + text)
                if isinstance(content, list):
                    text_parts = [
                        item.get("text", item.get("content", ""))
                        for item in content
                        if isinstance(item, dict)
                    ]
                    content = "".join(text_parts)
                if content:
                    chunk_count += 1
                    yield content

        # Track usage once from the last chunk that had it
        if last_chunk_with_usage:
            tracker.track(last_chunk_with_usage)
        elif chunk_count > 0:
            # No usage in any chunk — estimate from chunk count
            tracker.track_streaming([None] * chunk_count)

        self._request_context["token_usage"] = tracker

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

    @mlflow.trace(span_type="TOOL", name="execute_tools")
    def _execute_tools(self, user_message: str) -> str:
        """Run tools based on patterns in the user message. Returns tool context or empty string."""
        import json as _json
        tool_results = []

        # Calculate: detect math expressions
        calc_pattern = r'(?:calculate|compute|what is|how much is)\s+(.+?)(?:\?|$)'
        calc_match = re.search(calc_pattern, user_message.lower())
        if calc_match:
            expr = calc_match.group(1).strip()
            # Check if it looks like a math expression
            if re.match(r'^[\d\s\+\-\*\/\.\(\)%]+$', expr):
                result = calculate(expr)
                if "result" in result:
                    tool_results.append(f"[Calculator] {expr} = {result['result']}")

        # Timestamp: detect time-related queries
        if any(kw in user_message.lower() for kw in ["current time", "what time", "timestamp", "current date"]):
            ts = get_current_timestamp()
            tool_results.append(f"[Timestamp] Current time: {ts['utc']}")

        # SQL formatting: detect SQL in the query
        sql_pattern = r'(?:format|validate|check)(?:\s+this)?\s+sql[:\s]+(.+)'
        sql_match = re.search(sql_pattern, user_message.lower(), re.DOTALL)
        if sql_match:
            sql_text = sql_match.group(1).strip().strip('`"\'')
            result = format_sql(sql_text)
            if "formatted" in result:
                parts = [f"[SQL Formatter] {result['formatted']}"]
                if result.get("warnings"):
                    parts.append(f"Warnings: {'; '.join(result['warnings'])}")
                tool_results.append("\n".join(parts))

        # Cluster sizing: detect sizing/capacity queries
        msg_lower = user_message.lower()
        sizing_keywords = ["cluster size", "cluster sizing", "how many nodes", "how many workers",
                           "node type", "instance type", "cluster config", "cluster recommendation",
                           "capacity planning", "cluster for"]
        if any(kw in msg_lower for kw in sizing_keywords):
            # Extract data size if mentioned
            size_match = re.search(r'(\d+\.?\d*)\s*(gb|tb|mb|petabyte|pb)', msg_lower)
            data_gb = 100  # default
            if size_match:
                size_val = float(size_match.group(1))
                unit = size_match.group(2)
                if unit == "tb":
                    data_gb = size_val * 1024
                elif unit in ("pb", "petabyte"):
                    data_gb = size_val * 1024 * 1024
                elif unit == "mb":
                    data_gb = size_val / 1024
                else:
                    data_gb = size_val

            # Detect use case
            use_case = "etl"
            if any(kw in msg_lower for kw in ["ml", "training", "model", "machine learning"]):
                use_case = "ml_training"
            elif any(kw in msg_lower for kw in ["stream", "real-time", "realtime", "kafka"]):
                use_case = "streaming"
            elif any(kw in msg_lower for kw in ["sql", "analytics", "bi", "query", "dashboard"]):
                use_case = "sql_analytics"
            elif any(kw in msg_lower for kw in ["inference", "scoring", "prediction", "batch scoring"]):
                use_case = "ml_inference"

            # Detect cloud
            cloud = "AWS"
            if any(kw in msg_lower for kw in ["azure", "standard_"]):
                cloud = "Azure"

            # Detect specific node type
            node_type = None
            for nt in NODE_CATALOG:
                if nt.lower() in msg_lower:
                    node_type = nt
                    break

            result = cluster_sizing(data_gb, use_case, node_type, cloud)
            if "error" not in result:
                rec = result["recommendation"]
                cap = result["cluster_capacity"]
                sizing = result["sizing_breakdown"]
                parts = [
                    f"[Cluster Sizing] Use case: {result['use_case']} | Data: {data_gb} GB",
                    f"  Node type: {rec['node_type']} ({result['node_specs']['vcpus']} vCPUs, {result['node_specs']['memory_gb']} GB RAM)",
                    f"  Workers: {rec['num_workers']} (autoscale {rec['autoscale']['min_workers']}-{rec['autoscale']['max_workers']})",
                    f"  Total capacity: {cap['total_vcpus']} vCPUs, {cap['total_memory_gb']} GB RAM, {cap['total_dbu_per_hour']} DBU/hr",
                    f"  Limiting factor: {sizing['limiting_factor']}",
                    f"  Tips: {'; '.join(result['tips'])}",
                ]
                tool_results.append("\n".join(parts))

        # Node info: detect node type lookup
        node_pattern = r'(?:what is|specs for|info on|details of)\s+((?:i3|m5|r5|c5|p3|g5|Standard_)\S+)'
        node_match = re.search(node_pattern, user_message, re.IGNORECASE)
        if node_match:
            result = get_node_info(node_match.group(1))
            if "error" not in result:
                tool_results.append(
                    f"[Node Info] {result['node_type']}: {result['vcpus']} vCPUs, "
                    f"{result['memory_gb']} GB RAM, {result.get('storage_gb', 0)} GB storage, "
                    f"{result['dbu_per_hour']} DBU/hr ({result['category']})"
                )

        # Execute custom team tools
        if self.custom_tools:
            custom_results = execute_custom_tools(self.custom_tools, user_message)
            tool_results.extend(custom_results)

        if tool_results:
            return "Tool Results:\n" + "\n".join(tool_results)
        return ""

    def _build_augmented_messages(self, messages):
        """Build the augmented prompt (shared between predict and predict_stream)."""
        recent_user_msgs = [
            m.content for m in messages if m.role == "user"
        ][-self.max_retrieval_turns:]
        retrieval_query = " ".join(recent_user_msgs)

        context_text, retrieved_docs = self._retrieve_context(retrieval_query)

        augmented_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Documentation Context:\n\n{context_text}"},
        ]

        # Execute tools based on user message
        user_message = recent_user_msgs[-1] if recent_user_msgs else ""
        tool_context = self._execute_tools(user_message)
        if tool_context:
            augmented_messages.append({"role": "system", "content": tool_context})

        user_memories = self._request_context.get("user_memories", [])
        if user_memories:
            memory_lines = [f"  [{m['key']}]: {m['content']}" for m in user_memories]
            memory_text = "User context from prior sessions:\n" + "\n".join(memory_lines)
            augmented_messages.append({"role": "system", "content": memory_text})

        history = messages[-self.max_history_turns:] if len(messages) > self.max_history_turns else messages
        for msg in history:
            augmented_messages.append({"role": msg.role, "content": msg.content})

        return augmented_messages, retrieved_docs

    @mlflow.trace(span_type="CHAIN")
    def _process_request_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[dict] = None,
        custom_inputs: Optional[dict] = None,
    ):
        """Streaming version — yields ChatAgentChunk objects token by token."""
        from mlflow.types.agent import ChatAgentChunk

        augmented_messages, retrieved_docs = self._build_augmented_messages(messages)
        self._request_context["retrieved_docs"] = retrieved_docs

        msg_id = str(uuid.uuid4())
        for text in self._call_llm_stream(augmented_messages):
            yield ChatAgentChunk(
                delta=ChatAgentMessage(id=msg_id, role="assistant", content=text)
            )


# MLflow ChatAgent entry point
AGENT = DatabricksDocsAgent()
mlflow.models.set_model(AGENT)

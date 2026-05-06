"""
AgentOPS Databricks App — Async agent with Lakebase memory.

Adapts the AgentOPS framework (guardrails, tracing, session history, long-term memory)
to the Databricks Apps deployment pattern. Uses AsyncDatabricksStore and
AsyncCheckpointSaver for Lakebase connectivity as a first-class app resource.

Memory:
  - Short-term: per-thread conversation history via DatabricksStore
  - Long-term: cross-session user facts via DatabricksStore with semantic search
"""

import json
import logging
import os
import uuid
import time
import yaml
from typing import Any, Optional

import mlflow
from databricks_langchain import AsyncDatabricksStore
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ── MLflow ──
mlflow.set_tracking_uri("databricks")
_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
if _experiment_id:
    mlflow.set_experiment(experiment_id=_experiment_id)
    logger.info(f"MLflow configured: experiment_id={_experiment_id}")
elif _experiment_name:
    mlflow.set_experiment(experiment_name=_experiment_name)
    logger.info(f"MLflow configured: experiment_name={_experiment_name}")
else:
    # Auto-create experiment based on app name
    _auto_name = f"/Users/{os.getenv('DATABRICKS_CLIENT_ID', 'unknown')}/agentops-app"
    try:
        mlflow.set_experiment(experiment_name=_auto_name)
        logger.info(f"MLflow configured: auto-created experiment={_auto_name}")
    except Exception as e:
        logger.warning(f"MLflow experiment setup failed: {e}")

# ── Config from environment ──
LLM_ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "databricks-gpt-oss-120b")
CATALOG = os.getenv("CATALOG_NAME", "classic_stable_cykcbe_catalog")
SCHEMA = os.getenv("SCHEMA_NAME", "agentops")
VS_INDEX = os.getenv("VS_INDEX", f"{CATALOG}.{SCHEMA}.databricks_docs_index")
VS_NUM_RESULTS = int(os.getenv("VS_NUM_RESULTS", "5"))

LAKEBASE_PROJECT = os.getenv("LAKEBASE_AUTOSCALING_PROJECT", "agentops-sessions")
LAKEBASE_BRANCH = os.getenv("LAKEBASE_AUTOSCALING_BRANCH", "production")
LAKEBASE_INSTANCE = os.getenv("LAKEBASE_INSTANCE_NAME")  # For Provisioned mode

EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "1024"))

# UC Delta audit trail (via SQL Statement Execution API — no Spark needed)
UC_SESSION_TABLE = os.getenv("UC_SESSION_TABLE", f"{CATALOG}.{SCHEMA}.session_history")
UC_SESSION_ENABLED = os.getenv("UC_SESSION_ENABLED", "true").lower() == "true"
SQL_WAREHOUSE_ID = os.getenv("SQL_WAREHOUSE_ID", "")  # empty = auto-discover

# ── Load guardrail config ──
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
_config = {}
if os.path.exists(_config_path):
    with open(_config_path) as f:
        _config = yaml.safe_load(f) or {}

# Load system prompt: MLflow Prompt Registry → config.yaml fallback
_prompt_name = f"{CATALOG}.{SCHEMA}.databricks_docs_agent_system_prompt"
try:
    _prompt_obj = mlflow.genai.load_prompt(f"prompts:/{_prompt_name}@production")
    _base_prompt = _prompt_obj.template
    logger.info(f"Loaded prompt from registry: {_prompt_name}@production (v{_prompt_obj.version})")
except Exception as e:
    _base_prompt = _config.get("system_prompt", """You are a Databricks Documentation Assistant. You help users understand Databricks products, APIs, and best practices. Base your answers on the provided documentation context. Cite sources when referencing specific docs.""")
    logger.info(f"Prompt registry unavailable ({e}), using config.yaml")

MEMORY_PROMPT = """
You have long-term memory tools. Use them SILENTLY to personalize responses:
- When the user shares personal info (name, role, team, preferences, project details), call save_memory to store it for future sessions. Do NOT announce that you saved it — just acknowledge naturally.
- Do NOT call recall_memories proactively. The system already loads relevant memories for you (shown as "User context from prior sessions" above). Just use that context naturally in your answers.
- Do NOT start conversations with "Welcome back" or mention what you remember. Simply answer the question, using your knowledge of the user to tailor the response.
- Only save lasting user facts, not conversation content or questions.
"""

# Only append memory instructions when Lakebase is configured
_lakebase_configured = bool(LAKEBASE_PROJECT or LAKEBASE_INSTANCE)
SYSTEM_PROMPT = _base_prompt + MEMORY_PROMPT if _lakebase_configured else _base_prompt

# ── Guardrails (imported from framework if available, else inline) ──
try:
    from guardrails.pre_llm import PreLLMGuardrails
    from guardrails.post_llm import PostLLMGuardrails
    _gr_config = _config.get("guardrails", {})
    _pre_guardrails = PreLLMGuardrails(_gr_config.get("pre_llm", {}))
    _post_guardrails = PostLLMGuardrails(_gr_config.get("post_llm", {}))
    GUARDRAILS_ENABLED = _gr_config.get("enabled", True)
except ImportError:
    logger.warning("Guardrails not available — running without pre/post LLM checks")
    _pre_guardrails = None
    _post_guardrails = None
    GUARDRAILS_ENABLED = False

# ── Workspace client (singleton) ──
_ws_client: Optional[WorkspaceClient] = None


def _get_ws() -> WorkspaceClient:
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client


def _get_lakebase_kwargs() -> dict[str, Any]:
    """Lakebase connection kwargs — supports both Autoscaling and Provisioned."""
    if LAKEBASE_INSTANCE:
        return {"instance_name": LAKEBASE_INSTANCE}
    return {"project": LAKEBASE_PROJECT, "branch": LAKEBASE_BRANCH}


def _get_store_kwargs() -> dict[str, Any]:
    """DatabricksStore kwargs including embedding config for semantic search."""
    return {
        "embedding_endpoint": EMBEDDING_ENDPOINT,
        "embedding_dims": EMBEDDING_DIMS,
        **_get_lakebase_kwargs(),
    }


# ── UC Delta Audit Trail (via SQL Statement Execution API) ──

_warehouse_id_cache: str | None = None


def _resolve_warehouse() -> str | None:
    """Find a SQL warehouse for UC writes. Caches result."""
    global _warehouse_id_cache
    if _warehouse_id_cache is not None:
        return _warehouse_id_cache or None

    if SQL_WAREHOUSE_ID:
        _warehouse_id_cache = SQL_WAREHOUSE_ID
        return SQL_WAREHOUSE_ID

    try:
        w = _get_ws()
        for wh in w.warehouses.list():
            if wh.warehouse_type and "SERVERLESS" in str(wh.warehouse_type).upper():
                _warehouse_id_cache = wh.id
                logger.info(f"Auto-resolved serverless warehouse: {wh.name} ({wh.id})")
                return wh.id
        for wh in w.warehouses.list():
            if wh.state and "RUNNING" in str(wh.state).upper():
                _warehouse_id_cache = wh.id
                logger.info(f"Auto-resolved running warehouse: {wh.name} ({wh.id})")
                return wh.id
        _warehouse_id_cache = ""
        return None
    except Exception as e:
        logger.error(f"Warehouse discovery failed: {e}")
        _warehouse_id_cache = ""
        return None


import re as _re


def _validate_table_name(name: str) -> str:
    """Validate a fully-qualified table name to prevent SQL injection."""
    if not _re.match(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$', name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


_uc_table_ensured = False


def _exec_uc_sql(statement: str, parameters: list = None):
    """Execute SQL via Statement Execution API with parameterized query support."""
    from databricks.sdk.service.sql import StatementState, StatementParameterListItem

    wh_id = _resolve_warehouse()
    if not wh_id:
        raise RuntimeError("No SQL warehouse available")
    w = _get_ws()
    kwargs = {"warehouse_id": wh_id, "statement": statement}
    if parameters:
        # Convert dicts to StatementParameterListItem objects for newer SDK versions
        kwargs["parameters"] = [
            StatementParameterListItem(**p) if isinstance(p, dict) else p
            for p in parameters
        ]
    resp = w.statement_execution.execute_statement(**kwargs)
    state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        error_info = getattr(resp.status, 'error', None)
        raise RuntimeError(f"SQL failed: state={state}, error={error_info}")
    return resp


def _save_turn_to_uc(
    thread_id: str, turn_number: int, user_message: str,
    assistant_response: str, response_time_ms: float,
    model_endpoint: str = "", trace_id: str = "",
):
    """Append a turn to UC Delta table using parameterized SQL (no injection risk)."""
    global _uc_table_ensured
    if not UC_SESSION_ENABLED:
        return
    try:
        table = _validate_table_name(UC_SESSION_TABLE)

        # Ensure table exists (once per app lifetime, DDL uses validated identifier)
        if not _uc_table_ensured:
            _exec_uc_sql(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    turn_id STRING NOT NULL, session_id STRING NOT NULL,
                    turn_number INT NOT NULL, user_message STRING,
                    assistant_response STRING, request_time STRING NOT NULL,
                    response_time_ms DOUBLE, model_endpoint STRING,
                    trace_id STRING, metadata STRING
                )
            """)
            _uc_table_ensured = True

        turn_id = str(uuid.uuid4())
        request_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Parameterized INSERT — safe from SQL injection
        _exec_uc_sql(
            f"""INSERT INTO {table}
                (turn_id, session_id, turn_number, user_message, assistant_response,
                 request_time, response_time_ms, model_endpoint, trace_id, metadata)
                VALUES (:turn_id, :session_id, :turn_number, :user_message, :assistant_response,
                        :request_time, :response_time_ms, :model_endpoint, :trace_id, :metadata)""",
            parameters=[
                {"name": "turn_id", "value": turn_id, "type": "STRING"},
                {"name": "session_id", "value": thread_id, "type": "STRING"},
                {"name": "turn_number", "value": str(turn_number), "type": "INT"},
                {"name": "user_message", "value": user_message[:4000], "type": "STRING"},
                {"name": "assistant_response", "value": assistant_response[:4000], "type": "STRING"},
                {"name": "request_time", "value": request_time, "type": "STRING"},
                {"name": "response_time_ms", "value": str(round(response_time_ms, 2)), "type": "DOUBLE"},
                {"name": "model_endpoint", "value": model_endpoint, "type": "STRING"},
                {"name": "trace_id", "value": trace_id, "type": "STRING"},
                {"name": "metadata", "value": "", "type": "STRING"},
            ],
        )
        logger.info(f"UC(SQL API): saved turn session={thread_id}, turn={turn_number}")
    except Exception as e:
        logger.error(f"UC(SQL API) write failed: {e}")


# ── UC Delta Read (session history fallback) ──

def _read_history_from_uc(thread_id: str, max_turns: int = 20) -> list[dict]:
    """Read session history from UC Delta table using parameterized query.

    Used as fallback when Lakebase has no history for this thread.
    """
    if not UC_SESSION_ENABLED:
        return []
    try:
        table = _validate_table_name(UC_SESSION_TABLE)
        resp = _exec_uc_sql(
            f"""SELECT turn_number, user_message, assistant_response
                FROM {table}
                WHERE session_id = :session_id
                ORDER BY turn_number DESC
                LIMIT {int(max_turns)}""",
            parameters=[
                {"name": "session_id", "value": thread_id, "type": "STRING"},
            ],
        )
        if not resp.result or not resp.result.data_array:
            return []
        messages = []
        for row in reversed(resp.result.data_array):
            user_msg = row[1] if len(row) > 1 else ""
            asst_msg = row[2] if len(row) > 2 else ""
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if asst_msg:
                messages.append({"role": "assistant", "content": asst_msg})
        logger.info(f"UC(SQL API): read {len(resp.result.data_array)} turns for thread {thread_id}")
        return messages
    except Exception as e:
        logger.error(f"UC read failed: {e}")
        return []


# ── Vector Search ──
@mlflow.trace(name="search_docs", span_type="RETRIEVER")
def _search_docs(query: str) -> tuple[str, list[dict]]:
    """Retrieve relevant docs via vector search (SDK-based)."""
    try:
        w = _get_ws()
        results = w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            query_text=query,
            columns=["chunk_text", "url", "chunk_id"],
            num_results=VS_NUM_RESULTS,
        )
        docs = []
        if hasattr(results, "result") and results.result and results.result.data_array:
            for row in results.result.data_array:
                docs.append({
                    "chunk_text": row[0] if len(row) > 0 else "",
                    "url": row[1] if len(row) > 1 else "",
                    "chunk_id": row[2] if len(row) > 2 else "",
                })
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}] {doc.get('url', '')}\n{doc.get('chunk_text', '')}")
        return "\n\n---\n\n".join(context_parts), docs
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return "No relevant documentation found.", []


# ── Memory Tool Definitions ──
MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save an important fact about this user to long-term memory. Call this when the user shares personal preferences, their role, team, project details, or anything worth remembering across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short label for the fact, e.g. 'name', 'role', 'preferences', 'team', 'focus_area'",
                    },
                    "value": {
                        "type": "string",
                        "description": "The fact to remember, e.g. 'Data Engineer at Acme Corp', 'prefers Python over Scala'",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Search this user's long-term memory for relevant facts from prior sessions. Call this at the start of a conversation to personalize your response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, e.g. 'user preferences', 'what does the user work on'",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ── LLM Call (with tool support) ──
@mlflow.trace(name="call_llm", span_type="LLM")
def _call_llm(messages: list[dict], tools: list[dict] = None) -> dict:
    """Call the LLM endpoint. Returns {"content": str, "tool_calls": list|None}.

    Uses the SDK's api_client.do() for tool-calling support, which handles
    auth automatically in Databricks App context.
    """
    w = _get_ws()

    # Build request body — pass messages as plain dicts for tool compatibility
    body = {
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    if tools:
        body["tools"] = tools

    data = w.api_client.do(
        "POST",
        f"/serving-endpoints/{LLM_ENDPOINT}/invocations",
        body=body,
    )

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    if isinstance(content, list):
        text_parts = [
            item.get("text", item.get("content", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in ("text", "output_text", None)
        ]
        content = "\n".join(text_parts) if text_parts else str(content)

    tool_calls = msg.get("tool_calls")
    return {"content": content, "tool_calls": tool_calls}


# ── Main Agent Runner ──
async def run_agent(
    messages: list[dict],
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run the agent with Lakebase-backed memory.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."}
        thread_id: Thread ID for conversation continuity (omit for new conversation)
        user_id: User ID for long-term memory (optional)

    Returns:
        {"output": "response text", "thread_id": "...", "user_id": "..."}
    """
    start_time = time.time()
    if not thread_id:
        thread_id = str(uuid.uuid4())

    user_message = messages[-1].get("content", "") if messages else ""
    conversation_context = ""
    # Sanitize user_id for Lakebase namespace (no dots allowed)
    safe_user_id = user_id.replace(".", "_").replace("@", "_at_") if user_id else None

    prior_messages = []
    recalled_memories = []
    response_text = ""

    # Root span — all child spans nest under this single trace
    with mlflow.start_span(name="agentops_predict", span_type="AGENT") as root_span:
        root_span.set_inputs({"messages": messages, "thread_id": thread_id, "user_id": user_id or ""})

        # Try Lakebase store — gracefully degrade if unavailable
        _store_available = False
        store = None
        try:
            store = AsyncDatabricksStore(**_get_store_kwargs())
            await store.__aenter__()
            await store.setup()
            _store_available = True
        except Exception as e:
            logger.warning(f"Lakebase unavailable, running without memory: {e}")
            store = None

        try:

            # ── Load session history (Lakebase primary → UC fallback) ──
            with mlflow.start_span(name="load_session_history", span_type="RETRIEVER") as span:
                history_source = "none"
                if _store_available and store:
                    try:
                        item = await store.aget(("conversations",), thread_id)
                        prior_messages = item.value.get("messages", []) if item and item.value else []
                        if prior_messages:
                            history_source = "lakebase"
                    except Exception as e:
                        logger.error(f"Lakebase history load failed: {e}")
                        prior_messages = []

                # Fallback: read from UC Delta if Lakebase had nothing
                if not prior_messages and UC_SESSION_ENABLED:
                    uc_messages = _read_history_from_uc(thread_id)
                    if uc_messages:
                        prior_messages = uc_messages
                        history_source = "uc_delta"

                span.set_inputs({"thread_id": thread_id})
                span.set_outputs({
                    "source": history_source,
                    "turns_loaded": len(prior_messages),
                    "messages": prior_messages[-10:],  # Last 10 messages for readability
                })
                span.set_attributes({
                    "thread_id": thread_id,
                    "turns_loaded": len(prior_messages),
                    "source": history_source,
                })

            if prior_messages:
                conversation_context = " ".join(
                    m["content"] for m in prior_messages if m.get("role") == "user"
                )

            all_messages = prior_messages + messages

            # ── Recall long-term user memories ──
            # Done BEFORE guardrails so recalled context can inform intent check
            if user_id and _store_available and store:
                with mlflow.start_span(name="recall_user_memory", span_type="RETRIEVER") as span:
                    try:
                        results = await store.asearch(("users", safe_user_id), query=user_message, limit=5)
                        recalled_memories = [
                            {"key": r.key, "content": r.value.get("content", "")}
                            for r in results
                        ]
                        span.set_inputs({"user_id": user_id, "query": user_message[:100]})
                        span.set_outputs({
                            "memories_found": len(recalled_memories),
                            "memories": recalled_memories,
                        })
                        span.set_attributes({
                            "user_id": user_id, "query": user_message[:100],
                            "memories_found": len(recalled_memories),
                            "memory_keys": [m["key"] for m in recalled_memories],
                        })
                        # Enrich conversation_context with recalled memories for intent check
                        if recalled_memories and not conversation_context:
                            conversation_context = " ".join(
                                m["content"] for m in recalled_memories
                            )
                    except Exception as e:
                        logger.error(f"Memory recall failed: {e}")
                        span.set_attributes({"user_id": user_id, "memories_found": 0, "error": str(e)})

            # ── Pre-LLM Guardrails ──
            if GUARDRAILS_ENABLED and _pre_guardrails:
                pre_result = _pre_guardrails.check(
                    user_message, conversation_context=conversation_context)
                if pre_result.get("blocked"):
                    return {"output": pre_result.get("message", "Request blocked."),
                            "thread_id": thread_id, "user_id": user_id or ""}

            # ── Retrieve docs (vector search) ──
            recent_user_msgs = [m["content"] for m in all_messages if m.get("role") == "user"][-3:]
            retrieval_query = " ".join(recent_user_msgs)
            context_text, retrieved_docs = _search_docs(retrieval_query)

            # ── Build augmented prompt ──
            augmented = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Documentation Context:\n\n{context_text}"},
            ]
            if recalled_memories:
                memory_lines = [f"  [{m['key']}]: {m['content']}" for m in recalled_memories]
                augmented.append({
                    "role": "system",
                    "content": "User context from prior sessions:\n" + "\n".join(memory_lines),
                })
            augmented.extend(all_messages[-20:])

            # ── Call LLM with tool loop ──
            # The LLM can call save_memory/recall_memories tools.
            # We execute tool calls and loop until the LLM produces a final text response.
            use_tools = MEMORY_TOOLS if (user_id and _store_available) else None
            max_tool_rounds = 3
            for _round in range(max_tool_rounds + 1):
                llm_result = _call_llm(augmented, tools=use_tools)
                tool_calls = llm_result.get("tool_calls")

                if not tool_calls:
                    response_text = llm_result["content"]
                    break

                # Execute tool calls (tool_calls are raw dicts from REST API)
                with mlflow.start_span(name="execute_memory_tools", span_type="TOOL") as tspan:
                    tool_results = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        fn_name = fn.get("name", "")
                        fn_args_raw = fn.get("arguments", "{}")
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        tool_id = tc.get("id", "")

                        if fn_name == "save_memory" and user_id:
                            key, value = fn_args.get("key", ""), fn_args.get("value", "")
                            await store.aput(("users", safe_user_id), key, {"content": value})
                            result_text = f"Saved '{key}' = '{value}' to long-term memory."
                            logger.info(f"Memory saved: user={user_id} key={key}")
                        elif fn_name == "recall_memories" and user_id:
                            query = fn_args.get("query", user_message)
                            results = await store.asearch(("users", safe_user_id), query=query, limit=5)
                            if results:
                                lines = [f"  [{r.key}]: {r.value.get('content','')}" for r in results]
                                result_text = "Recalled memories:\n" + "\n".join(lines)
                            else:
                                result_text = "No memories found for this user."
                        else:
                            result_text = f"Unknown tool: {fn_name}"

                        tool_results.append({"tool_call_id": tool_id, "name": fn_name, "result": result_text})

                    tspan.set_attributes({
                        "tools_called": [t["name"] for t in tool_results],
                        "round": _round + 1,
                    })

                # Append assistant tool_call message + tool results to conversation
                augmented.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                for tr in tool_results:
                    augmented.append({"role": "tool", "tool_call_id": tr["tool_call_id"], "content": tr["result"]})
            else:
                # Exhausted tool rounds — use last content
                response_text = llm_result.get("content", "I was unable to complete my response.")

            # ── Post-LLM Guardrails ──
            if GUARDRAILS_ENABLED and _post_guardrails:
                post_result = _post_guardrails.check(
                    user_message, response_text,
                    {"retrieved_docs": retrieved_docs, "user_id": user_id or ""})
                if post_result.get("blocked"):
                    response_text = post_result.get("message", "Response filtered for safety.")

            # ── Save session history ──
            if _store_available and store:
              with mlflow.start_span(name="save_session_history", span_type="TOOL") as span:
                try:
                    updated_messages = list(all_messages) + [
                        {"role": "assistant", "content": response_text}
                    ]
                    await store.aput(
                        ("conversations",), thread_id,
                        {"messages": updated_messages,
                         "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "turn_count": sum(1 for m in updated_messages if m.get("role") == "user")},
                    )
                    span.set_inputs({"thread_id": thread_id})
                    span.set_outputs({
                        "messages_saved": len(updated_messages),
                        "backends": ["lakebase", "uc_delta"] if UC_SESSION_ENABLED else ["lakebase"],
                    })
                    span.set_attributes({"thread_id": thread_id, "messages_saved": len(updated_messages)})
                except Exception as e:
                    logger.error(f"Failed to save history: {e}")
                    span.set_attributes({"error": str(e)})

            # ── Save conversation summary to long-term memory ──
            # So users can ask "recap last conversation" on a new thread
            if user_id and safe_user_id and _store_available and store:
                try:
                    user_turns = [m["content"] for m in all_messages if m.get("role") == "user"]
                    last_topics = user_turns[-5:]  # last 5 user messages as summary
                    summary = (
                        f"Last conversation (thread {thread_id}, "
                        f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}): "
                        f"User asked about: {' | '.join(last_topics[:3])}."
                    )
                    if response_text:
                        summary += f" Last answer covered: {response_text[:200]}"
                    await store.aput(
                        ("users", safe_user_id), "last_conversation_summary",
                        {"content": summary},
                    )
                except Exception as e:
                    logger.warning(f"Failed to save conversation summary: {e}")

            # ── UC Delta audit trail (via SQL Statement API) ──
            if UC_SESSION_ENABLED:
                try:
                    turn_count = sum(1 for m in all_messages if m.get("role") == "user")
                    _save_turn_to_uc(
                        thread_id=thread_id,
                        turn_number=turn_count,
                        user_message=user_message,
                        assistant_response=response_text,
                        response_time_ms=(time.time() - start_time) * 1000,
                        model_endpoint=LLM_ENDPOINT,
                    )
                except Exception as e:
                    logger.warning(f"UC audit trail write failed: {e}")

        finally:
            # Clean up Lakebase store if it was opened
            if store:
                try:
                    await store.__aexit__(None, None, None)
                except Exception:
                    pass

        latency_ms = (time.time() - start_time) * 1000
        root_span.set_attributes({
            "agentops.latency_ms": round(latency_ms, 2),
            "agentops.session.thread_id": thread_id,
            "agentops.session.history_turns": len(prior_messages) // 2,
            "agentops.memory.recalled_count": len(recalled_memories),
        })
        root_span.set_outputs({"response": response_text[:300], "thread_id": thread_id})

    logger.info(f"Request completed: thread={thread_id} latency={latency_ms:.0f}ms history={len(prior_messages)//2} memories={len(recalled_memories)}")

    return {
        "output": response_text,
        "thread_id": thread_id,
        "user_id": user_id or "",
    }

"""
Session & Memory Store — Persists conversation history and long-term agent memory.

Backends:
  - Unity Catalog Delta table — append-only audit trail (Spark SQL)
  - Lakebase via DatabricksStore (databricks_langchain) — low-latency memory
    - Short-term: per-thread conversation history
    - Long-term: cross-session user memory with semantic search

Connection pattern follows Databricks Lakebase Autoscaling reference:
  DatabricksStore(project=..., branch=..., workspace_client=WorkspaceClient())
  — handles OAuth credential generation internally, no raw psycopg2 needed.

Configuration lives in config.yaml under session_history and long_term_memory.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Manages session history persistence and long-term user memory.

    Short-term (conversation history per thread):
      - Lakebase DatabricksStore: namespace ("conversations",), key=thread_id
      - UC Delta table: append-only audit trail

    Long-term (user memory across sessions):
      - Lakebase DatabricksStore: namespace ("users", user_id), semantic search

    Usage in AgentOPSBase:
        self.session_store = SessionStore(config)
        self.session_store.get_history(thread_id)
        self.session_store.save_full_session(thread_id, messages, response, ...)
        self.session_store.recall_user_memories(user_id, query)
        self.session_store.save_user_memory(user_id, key, value)
    """

    def __init__(self, config: dict):
        session_cfg = config.get("session_history", {})
        self.enabled = session_cfg.get("enabled", False)

        # Global catalog/schema from config (reused for UC table)
        self.catalog = config.get("catalog", "")
        self.schema = config.get("schema", "")

        # Unity Catalog backend (audit trail)
        uc_cfg = session_cfg.get("unity_catalog", {})
        self.uc_enabled = self.enabled and uc_cfg.get("enabled", False)
        self.uc_table_name = uc_cfg.get("table", "session_history")
        self.uc_full_table = f"{self.catalog}.{self.schema}.{self.uc_table_name}" if self.catalog else self.uc_table_name

        # Lakebase backend (DatabricksStore — short-term + long-term memory)
        lakebase_cfg = session_cfg.get("lakebase", {})
        self.lakebase_enabled = self.enabled and lakebase_cfg.get("enabled", False)
        self.lakebase_project = lakebase_cfg.get("project", "")
        self.lakebase_branch = lakebase_cfg.get("branch", "production")

        # Long-term memory config (uses same Lakebase store instance)
        memory_cfg = config.get("long_term_memory", {})
        self.memory_enabled = memory_cfg.get("enabled", False)

        # Lazy-initialized DatabricksStore
        self._store = None
        self._store_attempted = False

        if self.enabled:
            backends = []
            if self.uc_enabled:
                backends.append(f"UC({self.uc_full_table})")
            if self.lakebase_enabled:
                backends.append(f"Lakebase({self.lakebase_project}/{self.lakebase_branch})")
            if self.memory_enabled:
                backends.append("LongTermMemory")
            logger.info(f"Session store enabled: {', '.join(backends) or 'no backends configured'}")

    # ── Lakebase DatabricksStore ─────────────────────────────────────

    def _get_store(self):
        """Lazy-init DatabricksStore for Lakebase Autoscaling.

        Uses databricks_langchain which handles OAuth credential generation
        internally — no raw psycopg2 or REST API calls needed.
        """
        if self._store is not None:
            return self._store
        if self._store_attempted:
            return None  # Already tried and failed

        self._store_attempted = True
        try:
            from databricks_langchain import DatabricksStore
            from databricks.sdk import WorkspaceClient

            self._store = DatabricksStore(
                project=self.lakebase_project,
                branch=self.lakebase_branch,
                workspace_client=WorkspaceClient(),
            )
            self._store.setup()
            self._store_error = None
            logger.info(f"DatabricksStore ready: {self.lakebase_project}/{self.lakebase_branch}")
            return self._store
        except Exception as e:
            self._store_error = str(e)
            logger.error(f"DatabricksStore init failed: {e}")
            return None

    # ── Short-term: Conversation History (per thread) ────────────────

    def get_history(self, thread_id: str, max_turns: int = 10) -> list[dict]:
        """
        Read conversation history for a thread.

        Priority: Lakebase (low latency) -> UC Delta (fallback).
        Returns list of {"role": "user"/"assistant", "content": "..."} dicts
        suitable for prepending to the agent's message list.
        """
        if not self.enabled or not thread_id:
            return []

        # Try Lakebase first (sub-ms reads)
        if self.lakebase_enabled:
            history = self._read_conversation(thread_id, max_turns)
            if history:
                return history

        # Fallback to UC Delta
        if self.uc_enabled:
            return self._read_from_uc(thread_id, max_turns)

        return []

    def _read_conversation(self, thread_id: str, max_turns: int) -> list[dict]:
        """Read conversation history from Lakebase DatabricksStore."""
        store = self._get_store()
        if not store:
            return []

        try:
            item = store.get(("conversations",), thread_id)
            if not item or not item.value:
                return []

            messages = item.value.get("messages", [])
            # Return last N turns (each turn = user + assistant = 2 messages)
            max_messages = max_turns * 2
            if len(messages) > max_messages:
                messages = messages[-max_messages:]

            logger.info(f"Lakebase: read {len(messages)} messages for thread {thread_id}")
            return messages
        except Exception as e:
            logger.error(f"Lakebase read failed for thread {thread_id}: {e}")
            return []

    def _save_conversation(self, thread_id: str, messages: list, response_text: str):
        """Save full conversation state to Lakebase DatabricksStore."""
        store = self._get_store()
        if not store:
            return

        try:
            # Serialize messages to simple dicts (handles both ChatAgentMessage and dict)
            serialized = []
            for m in messages:
                role = _get_role(m)
                content = _get_content(m)
                if role and content:
                    serialized.append({"role": role, "content": content})

            if response_text:
                serialized.append({"role": "assistant", "content": response_text})

            store.put(
                ("conversations",),
                thread_id,
                {
                    "messages": serialized,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "turn_count": sum(1 for m in serialized if m["role"] == "user"),
                },
            )
            logger.info(f"Lakebase: saved {len(serialized)} messages for thread {thread_id}")
        except Exception as e:
            logger.error(f"Lakebase save failed for thread {thread_id}: {e}")

    # ── Long-term: User Memory (cross-session) ──────────────────────

    def save_user_memory(self, user_id: str, key: str, value: str):
        """Save a fact about the user to long-term Lakebase memory.

        Args:
            user_id: Unique user identifier
            key: Short label for the fact (e.g., 'name', 'preferences')
            value: The fact to remember
        """
        if not self.memory_enabled or not self.lakebase_enabled:
            return
        store = self._get_store()
        if not store:
            return
        try:
            store.put(
                ("users", user_id),
                key,
                {"content": value, "saved_at": datetime.now(timezone.utc).isoformat()},
            )
            logger.info(f"Memory saved: user={user_id}, key={key}")
        except Exception as e:
            logger.error(f"Memory save failed for user {user_id}: {e}")

    def recall_user_memories(self, user_id: str, query: str, limit: int = 5) -> list[dict]:
        """Search user's long-term memory using semantic search.

        Args:
            user_id: Unique user identifier
            query: Natural-language description of what to recall
            limit: Max number of memories to return

        Returns:
            List of {"key": "...", "content": "..."} dicts
        """
        if not self.memory_enabled or not self.lakebase_enabled:
            return []
        store = self._get_store()
        if not store:
            return []
        try:
            results = store.search(("users", user_id), query=query, limit=limit)
            memories = [
                {"key": item.key, "content": item.value.get("content", "")}
                for item in results
            ]
            logger.info(f"Recalled {len(memories)} memories for user {user_id}")
            return memories
        except Exception as e:
            logger.error(f"Memory recall failed for user {user_id}: {e}")
            return []

    # ── Read Methods (UC Delta) ──────────────────────────────────────

    def _read_from_uc(self, session_id: str, max_turns: int) -> list[dict]:
        """Read session history from UC Delta table via Spark SQL."""
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if not spark:
                return []

            df = spark.sql(f"""
                SELECT turn_number, user_message, assistant_response
                FROM {self.uc_full_table}
                WHERE session_id = '{session_id}'
                ORDER BY turn_number DESC
                LIMIT {max_turns}
            """).collect()

            # Build message list in chronological order
            messages = []
            for row in reversed(df):
                if row.user_message:
                    messages.append({"role": "user", "content": row.user_message})
                if row.assistant_response:
                    messages.append({"role": "assistant", "content": row.assistant_response})

            logger.info(f"Read {len(df)} turns from UC for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to read session from UC: {e}")
            return []

    # ── Write Methods ────────────────────────────────────────────────

    def save_turn(
        self,
        session_id: str,
        turn_number: int,
        user_message: str,
        assistant_response: str,
        response_time_ms: float,
        model_endpoint: str = "",
        trace_id: str = "",
        extra_metadata: Optional[dict] = None,
    ):
        """Save a single conversation turn to UC Delta (audit trail)."""
        if not self.enabled or not self.uc_enabled:
            return

        now = datetime.now(timezone.utc)
        turn_id = str(uuid.uuid4())

        record = {
            "turn_id": turn_id,
            "session_id": session_id,
            "turn_number": turn_number,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "request_time": now.isoformat(),
            "response_time_ms": round(response_time_ms, 2),
            "model_endpoint": model_endpoint,
            "trace_id": trace_id,
        }
        if extra_metadata:
            record["metadata"] = str(extra_metadata)

        self._save_to_uc(record)

    def save_full_session(
        self,
        session_id: str,
        messages: list,
        response_text: str,
        response_time_ms: float,
        model_endpoint: str = "",
        trace_id: str = "",
        extra_metadata: Optional[dict] = None,
    ):
        """
        Save conversation state to Lakebase + audit turn to UC Delta.

        Lakebase: stores full conversation (overwrite per thread_id)
        UC Delta: appends individual turn record (audit trail)
        """
        if not self.enabled:
            return

        # Lakebase: save full conversation state
        if self.lakebase_enabled:
            self._save_conversation(session_id, messages, response_text)

        # UC Delta: save individual turn (audit trail)
        if self.uc_enabled:
            user_turns = sum(1 for m in messages if _get_role(m) == "user")
            user_message = ""
            for m in reversed(messages):
                if _get_role(m) == "user":
                    user_message = _get_content(m)
                    break

            self.save_turn(
                session_id=session_id,
                turn_number=user_turns,
                user_message=user_message,
                assistant_response=response_text,
                response_time_ms=response_time_ms,
                model_endpoint=model_endpoint,
                trace_id=trace_id,
                extra_metadata=extra_metadata,
            )

    # ── Unity Catalog Backend ────────────────────────────────────────

    def _save_to_uc(self, record: dict):
        """Append a turn record to a Unity Catalog Delta table via Spark."""
        try:
            from pyspark.sql import SparkSession
            from pyspark.sql.types import (
                StructType, StructField, StringType, IntegerType, DoubleType,
            )

            spark = SparkSession.getActiveSession()
            if not spark:
                logger.warning("No active Spark session — skipping UC session save")
                return

            schema = StructType([
                StructField("turn_id", StringType(), False),
                StructField("session_id", StringType(), False),
                StructField("turn_number", IntegerType(), False),
                StructField("user_message", StringType(), True),
                StructField("assistant_response", StringType(), True),
                StructField("request_time", StringType(), False),
                StructField("response_time_ms", DoubleType(), True),
                StructField("model_endpoint", StringType(), True),
                StructField("trace_id", StringType(), True),
                StructField("metadata", StringType(), True),
            ])

            row = {
                "turn_id": record["turn_id"],
                "session_id": record["session_id"],
                "turn_number": record["turn_number"],
                "user_message": record.get("user_message", ""),
                "assistant_response": record.get("assistant_response", ""),
                "request_time": record["request_time"],
                "response_time_ms": record.get("response_time_ms", 0.0),
                "model_endpoint": record.get("model_endpoint", ""),
                "trace_id": record.get("trace_id", ""),
                "metadata": record.get("metadata", ""),
            }

            df = spark.createDataFrame([row], schema=schema)
            df.write.mode("append").option("mergeSchema", "true").saveAsTable(self.uc_full_table)
            logger.info(f"UC: saved turn session={record['session_id']}, turn={record['turn_number']}")

        except Exception as e:
            logger.error(f"Failed to save session to UC: {e}")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_role(msg) -> str:
    """Extract role from ChatAgentMessage or dict."""
    if hasattr(msg, "role"):
        return msg.role
    if isinstance(msg, dict):
        return msg.get("role", "")
    return ""


def _get_content(msg) -> str:
    """Extract content from ChatAgentMessage or dict."""
    if hasattr(msg, "content"):
        return msg.content
    if isinstance(msg, dict):
        return msg.get("content", "")
    return ""

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
        # Warehouse ID: env var > config > auto-discover
        import os
        self.warehouse_id = os.getenv("SQL_WAREHOUSE_ID") or uc_cfg.get("warehouse_id", "auto")

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
        self._resolved_warehouse_id = None  # Lazy-resolved SQL warehouse
        self._uc_last_error = None  # Last UC write/read error for diagnostics

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
        """Read session history from UC Delta table.

        Tries Spark SQL first (notebooks/clusters), falls back to
        SQL Statement Execution API (Model Serving, Apps, anywhere).
        """
        # Try Spark first (zero-latency if available)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                return self._read_from_uc_spark(spark, session_id, max_turns)
        except ImportError:
            pass

        # Fallback: SQL Statement Execution API (works from Model Serving/Apps)
        return self._read_from_uc_sql_api(session_id, max_turns)

    def _read_from_uc_spark(self, spark, session_id: str, max_turns: int) -> list[dict]:
        """Read via Spark SQL (notebook/cluster context)."""
        try:
            df = spark.sql(f"""
                SELECT turn_number, user_message, assistant_response
                FROM {self.uc_full_table}
                WHERE session_id = '{session_id}'
                ORDER BY turn_number DESC
                LIMIT {max_turns}
            """).collect()

            messages = []
            for row in reversed(df):
                if row.user_message:
                    messages.append({"role": "user", "content": row.user_message})
                if row.assistant_response:
                    messages.append({"role": "assistant", "content": row.assistant_response})

            logger.info(f"UC(Spark): read {len(df)} turns for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"UC(Spark) read failed: {e}")
            return []

    def _read_from_uc_sql_api(self, session_id: str, max_turns: int) -> list[dict]:
        """Read via SQL Statement Execution API (no Spark needed)."""
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()

            safe_id = session_id.replace("'", "''")
            resp = self._exec_sql(w, f"""
                SELECT turn_number, user_message, assistant_response
                FROM {self.uc_full_table}
                WHERE session_id = '{safe_id}'
                ORDER BY turn_number DESC
                LIMIT {max_turns}
            """)

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

            logger.info(f"UC(SQL API): read {len(resp.result.data_array)} turns for session {session_id}")
            return messages
        except Exception as e:
            self._uc_last_error = f"read: {e}"
            logger.error(f"UC(SQL API) read failed: {e}")
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
            try:
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
                # Don't overwrite — _save_to_uc_sql_api sets _uc_last_error directly
            except Exception as e:
                self._uc_last_error = f"save_full_session: {e}"
                logger.error(f"UC save failed in save_full_session: {e}")

    # ── Unity Catalog Backend ────────────────────────────────────────

    def _save_to_uc(self, record: dict):
        """Append a turn record to UC Delta table.

        Tries Spark first (notebooks/clusters), falls back to
        SQL Statement Execution API (Model Serving, Apps, anywhere).
        """
        # Try Spark first
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                self._uc_write_path = "spark"
                self._save_to_uc_spark(spark, record)
                return
        except ImportError:
            pass

        # Fallback: SQL Statement Execution API
        self._uc_write_path = "sql_api"
        self._save_to_uc_sql_api(record)

    def _save_to_uc_spark(self, spark, record: dict):
        """Write via Spark DataFrame (notebook/cluster context)."""
        try:
            from pyspark.sql.types import (
                StructType, StructField, StringType, IntegerType, DoubleType,
            )

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
            logger.info(f"UC(Spark): saved turn session={record['session_id']}, turn={record['turn_number']}")
        except Exception as e:
            logger.error(f"UC(Spark) write failed: {e}")

    def _exec_sql(self, w, statement: str) -> object:
        """Execute a SQL statement via SQL Statement Execution API with proper status checking."""
        from databricks.sdk.service.sql import StatementState
        from datetime import timedelta

        wh_id = self._resolve_warehouse_id()
        if not wh_id:
            raise RuntimeError("No SQL warehouse configured")

        # Omit wait_timeout to use SDK default (synchronous execution).
        # Explicit values like "30s" or timedelta fail on some SDK versions.
        resp = w.statement_execution.execute_statement(
            warehouse_id=wh_id,
            statement=statement,
        )

        state = resp.status.state if resp.status else None
        if state != StatementState.SUCCEEDED:
            error_info = getattr(resp.status, 'error', None)
            raise RuntimeError(
                f"SQL failed: state={state}, error={error_info}"
            )
        return resp

    def _save_to_uc_sql_api(self, record: dict):
        """Write via SQL Statement Execution API (no Spark needed).

        Creates the table if it doesn't exist, then INSERTs the turn record.
        Works from Model Serving (with declared resources), Databricks Apps,
        or any SDK-authenticated context.
        """
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()

            # Ensure table exists (once per session lifetime)
            if not getattr(self, "_uc_table_ensured", False):
                self._exec_sql(w, f"""
                    CREATE TABLE IF NOT EXISTS {self.uc_full_table} (
                        turn_id STRING NOT NULL,
                        session_id STRING NOT NULL,
                        turn_number INT NOT NULL,
                        user_message STRING,
                        assistant_response STRING,
                        request_time STRING NOT NULL,
                        response_time_ms DOUBLE,
                        model_endpoint STRING,
                        trace_id STRING,
                        metadata STRING
                    )
                """)
                self._uc_table_ensured = True

            def esc(s):
                return str(s).replace("'", "''") if s else ""

            stmt = f"""
                INSERT INTO {self.uc_full_table}
                (turn_id, session_id, turn_number, user_message, assistant_response,
                 request_time, response_time_ms, model_endpoint, trace_id, metadata)
                VALUES (
                    '{esc(record["turn_id"])}',
                    '{esc(record["session_id"])}',
                    {record["turn_number"]},
                    '{esc(record.get("user_message", ""))}',
                    '{esc(record.get("assistant_response", ""))}',
                    '{esc(record["request_time"])}',
                    {record.get("response_time_ms", 0.0)},
                    '{esc(record.get("model_endpoint", ""))}',
                    '{esc(record.get("trace_id", ""))}',
                    '{esc(record.get("metadata", ""))}'
                )
            """
            self._exec_sql(w, stmt)

            # Verify: read back immediately to confirm data landed
            safe_sid = record["session_id"].replace("'", "''")
            verify = self._exec_sql(w, f"""
                SELECT COUNT(*) as cnt FROM {self.uc_full_table}
                WHERE session_id = '{safe_sid}' AND turn_id = '{record["turn_id"]}'
            """)
            rows = verify.result.data_array if verify.result else []
            count = int(rows[0][0]) if rows else 0
            if count > 0:
                self._uc_last_error = None
                logger.info(f"UC(SQL API): saved+verified turn session={record['session_id']}, turn={record['turn_number']}")
            else:
                self._uc_last_error = f"write_unverified: INSERT SUCCEEDED but SELECT found 0 rows"
                logger.error(f"UC(SQL API): INSERT claimed SUCCEEDED but verification found 0 rows for session={record['session_id']}")
        except Exception as e:
            self._uc_last_error = f"write: {e}"
            logger.error(f"UC(SQL API) write failed: {e}")

    # ── Warehouse Resolution ────────────────────────────────────────

    def _resolve_warehouse_id(self) -> Optional[str]:
        """Resolve warehouse_id — use configured value, or auto-discover a serverless warehouse."""
        if self._resolved_warehouse_id is not None:
            return self._resolved_warehouse_id

        if self.warehouse_id and self.warehouse_id != "auto":
            self._resolved_warehouse_id = self.warehouse_id
            return self.warehouse_id

        # Auto-discover: find a running or serverless SQL warehouse
        try:
            from databricks.sdk import WorkspaceClient
            w = WorkspaceClient()
            warehouses = w.warehouses.list()
            for wh in warehouses:
                # Prefer serverless (instant startup, scale to zero)
                if wh.warehouse_type and "SERVERLESS" in str(wh.warehouse_type).upper():
                    self._resolved_warehouse_id = wh.id
                    logger.info(f"Auto-resolved serverless warehouse: {wh.name} ({wh.id})")
                    return wh.id
            # Fallback: any running warehouse
            for wh in w.warehouses.list():
                if wh.state and "RUNNING" in str(wh.state).upper():
                    self._resolved_warehouse_id = wh.id
                    logger.info(f"Auto-resolved running warehouse: {wh.name} ({wh.id})")
                    return wh.id
            logger.warning("No SQL warehouse found for UC session history")
            self._resolved_warehouse_id = ""
            return None
        except Exception as e:
            logger.error(f"Warehouse auto-discovery failed: {e}")
            self._resolved_warehouse_id = ""
            return None


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

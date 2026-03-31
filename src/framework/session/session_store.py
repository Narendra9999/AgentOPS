"""
Session History Store — Persists multi-turn conversation history.

Supports two backends (both can be enabled simultaneously):
  - Unity Catalog Delta table — append-only, queryable via SQL/Spark
  - PostgreSQL (Lakebase) — low-latency reads for serving-layer session lookup

Configuration lives in config.yaml under session_history.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Manages session history persistence to UC Delta table and/or PostgreSQL.

    Usage in AgentOPSBase:
        self.session_store = SessionStore(config)
        self.session_store.save_turn(session_id, messages, response, metadata)
    """

    def __init__(self, config: dict):
        session_cfg = config.get("session_history", {})
        self.enabled = session_cfg.get("enabled", False)

        # Global catalog/schema from config (reused for UC table)
        self.catalog = config.get("catalog", "")
        self.schema = config.get("schema", "")

        # Unity Catalog backend
        uc_cfg = session_cfg.get("unity_catalog", {})
        self.uc_enabled = self.enabled and uc_cfg.get("enabled", False)
        self.uc_table_name = uc_cfg.get("table", "session_history")
        self.uc_full_table = f"{self.catalog}.{self.schema}.{self.uc_table_name}" if self.catalog else self.uc_table_name

        # PostgreSQL backend
        pg_cfg = session_cfg.get("postgres", {})
        self.pg_enabled = self.enabled and pg_cfg.get("enabled", False)
        self.pg_secret_scope = pg_cfg.get("secret_scope", "agentops_secrets")
        self.pg_secret_prefix = pg_cfg.get("secret_key_prefix", "pg_session")
        self.pg_table = pg_cfg.get("table", "session_history")
        self._pg_conn = None

        if self.enabled:
            backends = []
            if self.uc_enabled:
                backends.append(f"UC({self.uc_full_table})")
            if self.pg_enabled:
                backends.append(f"PG({self.pg_table})")
            logger.info(f"Session history enabled: {', '.join(backends) or 'no backends configured'}")

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
        """Save a single conversation turn to all enabled backends."""
        if not self.enabled:
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

        if self.uc_enabled:
            self._save_to_uc(record)

        if self.pg_enabled:
            self._save_to_pg(record)

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
        Save a complete session snapshot — all message turns plus the latest response.
        Calculates turn_number from the message history.
        """
        if not self.enabled:
            return

        # Count existing user turns to determine turn number
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

    # ── Unity Catalog Backend ───────────────────────────────────────────────

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
            logger.info(f"Session turn saved to UC: {self.uc_full_table} (session={record['session_id']}, turn={record['turn_number']})")

        except Exception as e:
            logger.error(f"Failed to save session to UC: {e}")

    # ── PostgreSQL Backend ──────────────────────────────────────────────────

    def _get_pg_connection(self):
        """Get or create a PostgreSQL connection using Databricks secrets."""
        if self._pg_conn is not None:
            try:
                self._pg_conn.cursor().execute("SELECT 1")
                return self._pg_conn
            except Exception:
                self._pg_conn = None

        try:
            import psycopg2
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            prefix = self.pg_secret_prefix

            host = w.dbutils.secrets.get(scope=self.pg_secret_scope, key=f"{prefix}_host")
            port = w.dbutils.secrets.get(scope=self.pg_secret_scope, key=f"{prefix}_port")
            db = w.dbutils.secrets.get(scope=self.pg_secret_scope, key=f"{prefix}_db")
            user = w.dbutils.secrets.get(scope=self.pg_secret_scope, key=f"{prefix}_user")
            password = w.dbutils.secrets.get(scope=self.pg_secret_scope, key=f"{prefix}_password")

            self._pg_conn = psycopg2.connect(
                host=host, port=int(port), dbname=db,
                user=user, password=password,
                connect_timeout=10,
            )
            self._pg_conn.autocommit = True

            # Ensure table exists
            self._ensure_pg_table()
            return self._pg_conn

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return None

    def _ensure_pg_table(self):
        """Create the session_history table in PostgreSQL if it doesn't exist."""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.pg_table} (
            turn_id         TEXT PRIMARY KEY,
            session_id      TEXT NOT NULL,
            turn_number     INTEGER NOT NULL,
            user_message    TEXT,
            assistant_response TEXT,
            request_time    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            response_time_ms DOUBLE PRECISION,
            model_endpoint  TEXT,
            trace_id        TEXT,
            metadata        TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{self.pg_table}_session
            ON {self.pg_table} (session_id, turn_number);
        """
        try:
            conn = self._pg_conn
            with conn.cursor() as cur:
                cur.execute(ddl)
            logger.info(f"PostgreSQL table ensured: {self.pg_table}")
        except Exception as e:
            logger.error(f"Failed to create PG table: {e}")

    def _save_to_pg(self, record: dict):
        """Insert a turn record into PostgreSQL."""
        conn = self._get_pg_connection()
        if not conn:
            return

        try:
            sql = f"""
            INSERT INTO {self.pg_table}
                (turn_id, session_id, turn_number, user_message, assistant_response,
                 request_time, response_time_ms, model_endpoint, trace_id, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (turn_id) DO NOTHING
            """
            with conn.cursor() as cur:
                cur.execute(sql, (
                    record["turn_id"],
                    record["session_id"],
                    record["turn_number"],
                    record.get("user_message", ""),
                    record.get("assistant_response", ""),
                    record["request_time"],
                    record.get("response_time_ms", 0.0),
                    record.get("model_endpoint", ""),
                    record.get("trace_id", ""),
                    record.get("metadata", ""),
                ))
            logger.info(f"Session turn saved to PG: {self.pg_table} (session={record['session_id']}, turn={record['turn_number']})")

        except Exception as e:
            logger.error(f"Failed to save session to PG: {e}")


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

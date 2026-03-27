"""
AgentOPS Framework — Audit Logger
Tracks pipeline executions, individual steps, deployments, and guardrail events.

Tables:
  - pipeline_execution_log: One row per pipeline run
  - pipeline_step_log: One row per step within a pipeline run
  - deployment_events: Cross-environment promotion events
  - guardrail_audit_log: Guardrail block/pass events
"""

import uuid
import json
import logging
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


def _safe_json_dumps(obj):
    """JSON dumps that handles numpy types."""
    def default(o):
        try:
            import numpy as np
            if isinstance(o, (np.bool_, np.integer)):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return str(o)
    return json.dumps(obj, default=default)


# ──────────────────────────────────────────────────────────────
# Table DDLs
# ──────────────────────────────────────────────────────────────

def get_audit_ddls(catalog: str, audit_schema: str) -> dict:
    p = f"{catalog}.{audit_schema}"
    return {
        "pipeline_execution_log": f"""
            CREATE TABLE IF NOT EXISTS {p}.pipeline_execution_log (
                execution_id STRING NOT NULL,
                pipeline_name STRING NOT NULL,
                agent_name STRING,
                environment STRING NOT NULL,
                triggered_by STRING,
                trigger_source STRING,
                depends_on STRING,
                status STRING,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds DOUBLE,
                error_message STRING
            )""",
        "pipeline_step_log": f"""
            CREATE TABLE IF NOT EXISTS {p}.pipeline_step_log (
                step_id STRING NOT NULL,
                execution_id STRING NOT NULL,
                step_name STRING NOT NULL,
                step_order INT,
                step_type STRING,
                depends_on STRING,
                status STRING,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds DOUBLE,
                records_processed BIGINT,
                input_params STRING,
                output_summary STRING,
                error_message STRING,
                databricks_job_run_id STRING
            )""",
        "deployment_events": f"""
            CREATE TABLE IF NOT EXISTS {p}.deployment_events (
                event_id STRING NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                agent_name STRING NOT NULL,
                agent_version STRING,
                source_environment STRING,
                target_environment STRING NOT NULL,
                deployed_by STRING,
                approved_by STRING,
                evaluation_passed BOOLEAN,
                evaluation_run_id STRING,
                jenkins_build_id STRING,
                commit_hash STRING,
                deployment_status STRING,
                error_message STRING
            )""",
        "guardrail_audit_log": f"""
            CREATE TABLE IF NOT EXISTS {p}.guardrail_audit_log (
                audit_id STRING NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                agent_name STRING NOT NULL,
                environment STRING NOT NULL,
                guardrail_type STRING,
                check_name STRING,
                blocked BOOLEAN,
                input_snippet STRING,
                reason STRING
            )""",
        "eval_results": f"""
            CREATE TABLE IF NOT EXISTS {p}.eval_results (
                evaluation_id STRING NOT NULL,
                execution_id STRING,
                row_index BIGINT,
                request STRING,
                response STRING,
                expected_response STRING,
                context STRING,
                toxicity_score DOUBLE,
                accuracy_score DOUBLE,
                helpfulness_score DOUBLE,
                professionalism_score DOUBLE,
                docs_relevance_score DOUBLE,
                code_snippet_score DOUBLE,
                source_citation_score DOUBLE,
                answer_completeness_score DOUBLE,
                overall_passed BOOLEAN,
                agent_name STRING,
                agent_version STRING,
                environment STRING,
                evaluated_at TIMESTAMP
            )""",
    }


def create_audit_tables(catalog: str, audit_schema: str, warehouse_id: str = None):
    """Create all audit tables. Run once per environment."""
    w = WorkspaceClient()
    if warehouse_id is None:
        warehouses = list(w.warehouses.list())
        warehouse_id = warehouses[0].id if warehouses else None

    # Create audit schema first
    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=f"CREATE SCHEMA IF NOT EXISTS {catalog}.{audit_schema}",
    )

    for table_name, ddl in get_audit_ddls(catalog, audit_schema).items():
        try:
            w.statement_execution.execute_statement(warehouse_id=warehouse_id, statement=ddl)
            logger.info(f"Created: {catalog}.{audit_schema}.{table_name}")
        except Exception as e:
            logger.error(f"Failed to create {table_name}: {e}")


# ──────────────────────────────────────────────────────────────
# Pipeline Step Logger — use this in notebooks
# ──────────────────────────────────────────────────────────────

class PipelineStepLogger:
    """
    Tracks pipeline execution and individual steps.

    Usage in a notebook:
        pipeline = PipelineStepLogger(catalog, audit_schema, "data_preparation", "my_agent", "dev")
        pipeline.start()

        step = pipeline.start_step("data_ingestion", step_order=1, step_type="data_prep")
        # ... do work ...
        pipeline.end_step(step, status="COMPLETED", records_processed=5000)

        pipeline.end(status="COMPLETED")
    """

    def __init__(self, catalog: str, audit_schema: str, pipeline_name: str,
                 agent_name: str = None, environment: str = "dev",
                 triggered_by: str = "manual", trigger_source: str = "",
                 depends_on: str = "", spark=None):
        self.catalog = catalog
        self.audit_schema = audit_schema
        self.pipeline_name = pipeline_name
        self.agent_name = agent_name or ""
        self.environment = environment
        self.triggered_by = triggered_by
        self.trigger_source = trigger_source
        self.depends_on = depends_on
        self.execution_id = str(uuid.uuid4())
        self.start_time = None
        self._spark = spark

    def _get_spark(self):
        """Get SparkSession — prefer passed-in spark, then try to get from environment."""
        if self._spark is not None:
            return self._spark
        try:
            from pyspark.sql import SparkSession
            self._spark = SparkSession.getActiveSession()
            return self._spark
        except Exception:
            pass
        return None

    def _execute(self, sql: str):
        """Execute SQL using spark.sql() (works on clusters without a SQL warehouse)."""
        try:
            spark = self._get_spark()
            if spark:
                spark.sql(sql)
            else:
                # Fallback to SDK statement execution
                w = WorkspaceClient()
                warehouses = list(w.warehouses.list())
                wh_id = warehouses[0].id if warehouses else None
                if wh_id:
                    w.statement_execution.execute_statement(
                        warehouse_id=wh_id, statement=sql)
                else:
                    logger.error("No spark session or SQL warehouse available for audit logging")
        except Exception as e:
            logger.error(f"Audit log SQL failed: {e}")

    def _ensure_tables(self):
        """Create audit schema + tables if they don't exist.
        Drops and recreates tables if schema doesn't match (migration)."""
        try:
            self._execute(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.audit_schema}")
            ddls = get_audit_ddls(self.catalog, self.audit_schema)
            for _tname, _ddl in ddls.items():
                fq = f"{self.catalog}.{self.audit_schema}.{_tname}"
                try:
                    # Check if table exists and has correct schema
                    spark = self._get_spark()
                    if spark:
                        try:
                            existing_cols = set(c.name for c in spark.table(fq).schema)
                            # Extract expected columns from DDL
                            import re
                            expected_cols = set(re.findall(r'(\w+)\s+(?:STRING|INT|BIGINT|DOUBLE|BOOLEAN|TIMESTAMP)', _ddl))
                            if not expected_cols.issubset(existing_cols):
                                missing = expected_cols - existing_cols
                                logger.warning(f"Schema mismatch in {fq}: missing {missing}. Recreating.")
                                self._execute(f"DROP TABLE IF EXISTS {fq}")
                                self._execute(_ddl)
                            # else: table exists with correct columns, skip
                        except Exception:
                            # Table doesn't exist, create it
                            self._execute(_ddl)
                    else:
                        self._execute(_ddl)
                except Exception as e:
                    logger.warning(f"Table {fq}: {e}")
                    try:
                        self._execute(f"DROP TABLE IF EXISTS {fq}")
                        self._execute(_ddl)
                    except Exception as e2:
                        logger.error(f"Failed to recreate {fq}: {e2}")
        except Exception as e:
            logger.error(f"Failed to ensure audit tables: {e}")

    def start(self):
        """Log pipeline start."""
        self._ensure_tables()
        self.start_time = datetime.now(timezone.utc)
        prefix = f"{self.catalog}.{self.audit_schema}"
        self._execute(f"""
            INSERT INTO {prefix}.pipeline_execution_log
            (execution_id, pipeline_name, agent_name, environment,
             triggered_by, trigger_source, depends_on, status, start_time)
            VALUES (
                '{self.execution_id}', '{self.pipeline_name}',
                '{self.agent_name}', '{self.environment}',
                '{self.triggered_by}', '{self.trigger_source}',
                '{self.depends_on}', 'RUNNING', current_timestamp()
            )""")
        logger.info(f"Pipeline started: {self.pipeline_name} [{self.execution_id}]")
        return self.execution_id

    def end(self, status: str = "COMPLETED", error_message: str = ""):
        """Log pipeline end."""
        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        prefix = f"{self.catalog}.{self.audit_schema}"
        self._execute(f"""
            UPDATE {prefix}.pipeline_execution_log
            SET status = '{status}',
                end_time = current_timestamp(),
                duration_seconds = {duration},
                error_message = '{error_message}'
            WHERE execution_id = '{self.execution_id}'
        """)
        logger.info(f"Pipeline {status}: {self.pipeline_name} [{self.execution_id}] ({duration:.1f}s)")

    def start_step(self, step_name: str, step_order: int = 0,
                   step_type: str = "", depends_on: str = "",
                   input_params: dict = None) -> dict:
        """Log step start. Returns step dict to pass to end_step()."""
        step_id = str(uuid.uuid4())
        params_json = _safe_json_dumps(input_params or {}).replace("'", "''")
        prefix = f"{self.catalog}.{self.audit_schema}"
        self._execute(f"""
            INSERT INTO {prefix}.pipeline_step_log
            (step_id, execution_id, step_name, step_order, step_type,
             depends_on, status, start_time, input_params)
            VALUES (
                '{step_id}', '{self.execution_id}', '{step_name}',
                {step_order}, '{step_type}', '{depends_on}', 'RUNNING',
                current_timestamp(), '{params_json}'
            )""")
        logger.info(f"  Step started: {step_name} [{step_id}]")
        return {"step_id": step_id, "step_name": step_name, "start": datetime.now(timezone.utc)}

    def end_step(self, step: dict, status: str = "COMPLETED",
                 records_processed: int = None, output_summary: dict = None,
                 error_message: str = "", job_run_id: str = ""):
        """Log step end."""
        duration = (datetime.now(timezone.utc) - step["start"]).total_seconds()
        output_json = _safe_json_dumps(output_summary or {}).replace("'", "''")
        records = records_processed if records_processed is not None else "NULL"
        prefix = f"{self.catalog}.{self.audit_schema}"
        self._execute(f"""
            UPDATE {prefix}.pipeline_step_log
            SET status = '{status}',
                end_time = current_timestamp(),
                duration_seconds = {duration},
                records_processed = {records},
                output_summary = '{output_json}',
                error_message = '{error_message}',
                databricks_job_run_id = '{job_run_id}'
            WHERE step_id = '{step["step_id"]}'
        """)
        logger.info(f"  Step {status}: {step['step_name']} ({duration:.1f}s, {records} records)")

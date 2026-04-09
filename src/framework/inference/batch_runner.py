"""
AgentOPS Framework — Batch Inference Runner
Processes a table of input queries through the agent endpoint using ai_query().
Separates successful responses from guardrail-blocked records (quarantine).

Flow:
  batch_input table → ai_query(endpoint) → batch_output table
                                         → batch_quarantine table (blocked by guardrails)
"""

import re
import mlflow
from databricks.sdk import WorkspaceClient
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier (catalog.schema.table) to prevent injection."""
    if not re.match(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$', name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


class BatchInferenceRunner:
    """
    Runs batch inference against a model serving endpoint.
    Uses ai_query() SQL function for efficient batch processing.
    Guardrail-blocked records go to a separate quarantine table.
    """

    def __init__(self, endpoint_name: str, catalog: str, schema: str):
        self.endpoint_name = endpoint_name
        self.catalog = catalog
        self.schema = schema
        self.w = WorkspaceClient()
        self._wh_id = None

    def _get_wh(self) -> str:
        if self._wh_id is None:
            warehouses = list(self.w.warehouses.list())
            self._wh_id = warehouses[0].id if warehouses else None
        return self._wh_id

    def _execute_sql(self, sql: str, timeout: str = "600s"):
        return self.w.statement_execution.execute_statement(
            warehouse_id=self._get_wh(),
            statement=sql,
            wait_timeout=timeout,
        )

    @mlflow.trace(span_type="BATCH_INFERENCE")
    def run_batch(
        self,
        input_table: str,
        output_table: str,
        quarantine_table: str,
        query_column: str = "question",
    ) -> dict:
        """
        Run batch inference on all rows in the input table.

        Args:
            input_table: Fully qualified table with input queries
            output_table: Fully qualified table for results
            quarantine_table: Fully qualified table for blocked records
            query_column: Column name containing the query text

        Returns:
            dict with status, counts, and duration
        """
        start = datetime.now(timezone.utc)

        fq_input = _validate_identifier(f"{self.catalog}.{self.schema}.{input_table}")
        fq_output = _validate_identifier(f"{self.catalog}.{self.schema}.{output_table}")
        fq_quarantine = _validate_identifier(f"{self.catalog}.{self.schema}.{quarantine_table}")

        try:
            # Step 1: Count input records
            result = self._execute_sql(f"SELECT count(*) FROM {fq_input}")
            input_count = int(result.result.data_array[0][0]) if result.result else 0
            logger.info(f"Batch input: {input_count} records from {fq_input}")

            if input_count == 0:
                return {
                    "status": "COMPLETED",
                    "input_records": 0,
                    "output_records": 0,
                    "quarantined_records": 0,
                    "duration_seconds": 0,
                }

            # Step 2: Run ai_query() on all input records
            # Validate identifiers to prevent injection (values come from config, not user input)
            _validate_identifier(query_column)
            _validate_identifier(self.endpoint_name)
            self._execute_sql(f"""
                CREATE OR REPLACE TABLE {fq_output} AS
                SELECT
                    input.*,
                    ai_query(
                        '{self.endpoint_name}',
                        input.{query_column}
                    ) as agent_response,
                    current_timestamp() as inference_timestamp,
                    'COMPLETED' as inference_status
                FROM {fq_input} input
            """)

            # Step 3: Count output records
            result = self._execute_sql(f"SELECT count(*) FROM {fq_output}")
            output_count = int(result.result.data_array[0][0]) if result.result else 0

            # Step 4: Identify and move quarantined records
            # Records where guardrails blocked the response
            self._execute_sql(f"""
                CREATE OR REPLACE TABLE {fq_quarantine} AS
                SELECT *
                FROM {fq_output}
                WHERE lower(agent_response) LIKE '%blocked%'
                   OR lower(agent_response) LIKE '%safety%filter%'
                   OR lower(agent_response) LIKE '%flagged by our safety%'
                   OR lower(agent_response) LIKE '%could not be processed%'
                   OR lower(agent_response) LIKE '%personal information%'
            """)

            result = self._execute_sql(f"SELECT count(*) FROM {fq_quarantine}")
            quarantine_count = int(result.result.data_array[0][0]) if result.result else 0

            duration = (datetime.now(timezone.utc) - start).total_seconds()

            logger.info(
                f"Batch complete: {output_count} output, {quarantine_count} quarantined "
                f"({duration:.1f}s)"
            )

            return {
                "status": "COMPLETED",
                "input_records": input_count,
                "output_records": output_count,
                "quarantined_records": quarantine_count,
                "duration_seconds": round(duration, 1),
                "output_table": fq_output,
                "quarantine_table": fq_quarantine,
            }

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            logger.error(f"Batch inference failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
                "duration_seconds": round(duration, 1),
            }

"""
AgentOPS Framework — Post-Deployment Trace Monitoring
Collects operational metrics, guardrail stats, and drift from production traces.
"""

import mlflow
from databricks.sdk import WorkspaceClient
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class TraceMonitor:
    """Monitors deployed agent traces for quality, drift, and guardrail performance."""

    def __init__(self, agent_name: str, environment: str, catalog: str = None,
                 schema: str = None, endpoint_name: str = None):
        self.agent_name = agent_name
        self.environment = environment
        self.catalog = catalog
        self.schema = schema
        self.endpoint_name = endpoint_name
        self.w = WorkspaceClient()

    def _get_warehouse_id(self) -> str:
        warehouses = list(self.w.warehouses.list())
        return warehouses[0].id if warehouses else None

    def collect_operational_metrics(self, hours: int = 1) -> dict:
        """Collect latency, error rates, token usage from inference table."""
        if not self.endpoint_name or not self.catalog or not self.schema:
            return {}

        wh_id = self._get_warehouse_id()
        try:
            # Inference table is auto-created by model serving auto-capture
            payload_table = f"{self.catalog}.{self.schema}.`{self.endpoint_name}_payload`"
            result = self.w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=f"""
                SELECT
                    count(*) as total_requests,
                    round(avg(response_duration_ms), 2) as avg_latency_ms,
                    round(percentile(response_duration_ms, 0.50), 2) as p50_latency_ms,
                    round(percentile(response_duration_ms, 0.95), 2) as p95_latency_ms,
                    round(percentile(response_duration_ms, 0.99), 2) as p99_latency_ms,
                    sum(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as error_count,
                    round(avg(total_tokens_count), 0) as avg_tokens_per_request
                FROM {payload_table}
                WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL {hours} HOURS) * 1000
                """)
            if result and result.result and result.result.data_array:
                columns = [col.name for col in result.manifest.schema.columns]
                row = result.result.data_array[0]
                return dict(zip(columns, row))
        except Exception as e:
            logger.warning(f"Failed to collect operational metrics: {e}")
        return {}

    def collect_guardrail_metrics(self, hours: int = 24) -> dict:
        """Collect guardrail block rates from MLflow traces."""
        try:
            cutoff = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp() * 1000)
            traces = mlflow.search_traces(
                experiment_names=[f"/agentops/{self.agent_name}"],
                filter_string=f"timestamp > {cutoff}",
                max_results=1000)

            metrics = {
                "total_requests": len(traces) if traces is not None else 0,
                "pre_llm_blocks": 0,
                "post_llm_blocks": 0,
                "block_reasons": {},
            }

            if traces is not None:
                for trace in traces.itertuples():
                    tags = getattr(trace, "tags", {}) or {}
                    if tags.get("agentops.guardrail.pre_llm.blocked") == "true":
                        metrics["pre_llm_blocks"] += 1
                        reason = tags.get("agentops.guardrail.pre_llm.blocked_by", "unknown")
                        metrics["block_reasons"][reason] = metrics["block_reasons"].get(reason, 0) + 1
                    if tags.get("agentops.guardrail.post_llm.blocked") == "true":
                        metrics["post_llm_blocks"] += 1

            if metrics["total_requests"] > 0:
                metrics["pre_llm_block_rate"] = metrics["pre_llm_blocks"] / metrics["total_requests"]
                metrics["post_llm_block_rate"] = metrics["post_llm_blocks"] / metrics["total_requests"]

            return metrics
        except Exception as e:
            logger.warning(f"Failed to collect guardrail metrics: {e}")
            return {}

    def check_quality_drift(self, window_days: int = 7) -> dict:
        """
        Compare recent trace quality vs baseline to detect drift.
        Checks: latency drift, token usage drift, error rate changes.
        """
        if not self.endpoint_name or not self.catalog or not self.schema:
            return {}

        wh_id = self._get_warehouse_id()
        try:
            payload_table = f"{self.catalog}.{self.schema}.`{self.endpoint_name}_payload`"
            result = self.w.statement_execution.execute_statement(
                warehouse_id=wh_id,
                statement=f"""
                WITH recent AS (
                    SELECT avg(response_duration_ms) as avg_latency,
                           avg(total_tokens_count) as avg_tokens
                    FROM {payload_table}
                    WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL 1 DAY) * 1000
                ),
                baseline AS (
                    SELECT avg(response_duration_ms) as avg_latency,
                           avg(total_tokens_count) as avg_tokens
                    FROM {payload_table}
                    WHERE timestamp_ms > unix_timestamp(current_timestamp() - INTERVAL {window_days} DAYS) * 1000
                      AND timestamp_ms <= unix_timestamp(current_timestamp() - INTERVAL 1 DAY) * 1000
                )
                SELECT
                    recent.avg_latency as recent_latency,
                    baseline.avg_latency as baseline_latency,
                    abs(recent.avg_latency - baseline.avg_latency) / nullif(baseline.avg_latency, 0) as latency_drift_pct,
                    recent.avg_tokens as recent_tokens,
                    baseline.avg_tokens as baseline_tokens,
                    abs(recent.avg_tokens - baseline.avg_tokens) / nullif(baseline.avg_tokens, 0) as token_drift_pct
                FROM recent, baseline
                """)
            if result and result.result and result.result.data_array:
                columns = [col.name for col in result.manifest.schema.columns]
                row = result.result.data_array[0]
                drift = dict(zip(columns, row))
                drift["latency_alert"] = float(drift.get("latency_drift_pct", 0) or 0) > 0.50
                drift["token_alert"] = float(drift.get("token_drift_pct", 0) or 0) > 0.25
                return drift
        except Exception as e:
            logger.warning(f"Failed to check drift: {e}")
        return {}

    def run_monitoring_cycle(self) -> dict:
        """Run a complete monitoring cycle."""
        logger.info(f"Monitoring: {self.agent_name} in {self.environment}")
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": self.agent_name,
            "environment": self.environment,
            "operational_metrics": self.collect_operational_metrics(hours=1),
            "guardrail_metrics": self.collect_guardrail_metrics(hours=24),
            "drift_check": self.check_quality_drift(window_days=7),
        }

# Databricks notebook source
# MAGIC %md
# MAGIC # Post-Deployment Monitoring
# MAGIC Collects operational metrics, guardrail stats, and drift detection.
# MAGIC Also ensures Lakehouse Monitor exists on the inference table.
# MAGIC Runs on schedule (hourly) via the monitoring job.

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")

# Find the deployed endpoint by prefix (handles truncated names)
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
match_prefix = f"agents_{catalog}-{schema}"
endpoint_name = None
for ep in w.serving_endpoints.list():
    if ep.name.startswith(match_prefix) and ep.state and str(ep.state.ready).endswith("READY"):
        endpoint_name = ep.name
        break

if not endpoint_name:
    print(f"No READY endpoint found matching '{match_prefix}...'")
    dbutils.notebook.exit('{"status": "SKIPPED", "reason": "no endpoint"}')

print(f"Monitoring endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Ensure Lakehouse Monitor exists on inference table

# COMMAND ----------

# Find the inference table (auto-created by AI Gateway)
_inference_tables = [t.name for t in spark.catalog.listTables(f"{catalog}.{schema}")
                     if agent_name.replace("_", "") in t.name.replace("_", "") and "payload" in t.name]
if _inference_tables:
    # Use the latest (highest suffix number)
    _inference_table = f"{catalog}.{schema}.{sorted(_inference_tables)[-1]}"
    print(f"Inference table: {_inference_table}")

    # Check if Lakehouse Monitor exists, create if not
    try:
        _existing_monitor = w.quality_monitors.get(table_name=_inference_table)
        print(f"Lakehouse Monitor exists: status={_existing_monitor.status}")
    except Exception:
        print("Creating Lakehouse Monitor...")
        try:
            from databricks.sdk.service.catalog import MonitorInferenceLog
            _monitor = w.quality_monitors.create(
                table_name=_inference_table,
                inference_log=MonitorInferenceLog(
                    problem_type="text",
                    prediction_col="response",
                    timestamp_col="timestamp_ms",
                    granularities=["1 day"],
                ),
                output_schema_name=f"{catalog}.{schema}",
            )
            print(f"Lakehouse Monitor created! Dashboard: {_monitor.dashboard_id}")
        except Exception as e:
            print(f"Lakehouse Monitor creation skipped: {e}")
            # Try time series fallback
            try:
                from databricks.sdk.service.catalog import MonitorTimeSeries
                _monitor = w.quality_monitors.create(
                    table_name=_inference_table,
                    time_series=MonitorTimeSeries(
                        timestamp_col="timestamp_ms",
                        granularities=["1 day"],
                    ),
                    output_schema_name=f"{catalog}.{schema}",
                )
                print(f"Time series monitor created! Dashboard: {_monitor.dashboard_id}")
            except Exception as e2:
                print(f"Time series monitor also failed: {e2}")
else:
    _inference_table = None
    print("No inference table found yet — Lakehouse Monitor will be created on next run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Run monitoring cycle

# COMMAND ----------

from framework.monitoring.trace_monitor import TraceMonitor

monitor = TraceMonitor(
    agent_name=agent_name,
    environment="dev",
    catalog=catalog,
    schema=schema,
    endpoint_name=endpoint_name,
)

results = monitor.run_monitoring_cycle()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Display results

# COMMAND ----------

import json
print("=== Operational Metrics (last 1 hour) ===")
print(json.dumps(results.get("operational_metrics", {}), indent=2, default=str))

print("\n=== Guardrail Metrics (last 24 hours) ===")
print(json.dumps(results.get("guardrail_metrics", {}), indent=2, default=str))

print("\n=== Drift Check (7-day window) ===")
drift = results.get("drift_check", {})
print(json.dumps(drift, indent=2, default=str))
if drift.get("latency_alert"):
    print("ALERT: Latency drift > 50%")
if drift.get("token_alert"):
    print("ALERT: Token usage drift > 25%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log to audit

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="monitoring", agent_name=agent_name, environment="dev",
    triggered_by="schedule", depends_on="none", spark=spark,
)
pipeline.start()

step = pipeline.start_step("ensure_lakehouse_monitor", step_order=1, step_type="monitoring", depends_on="none")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "inference_table": _inference_table or "not found",
    "monitor_exists": _inference_table is not None,
})

step = pipeline.start_step("collect_metrics", step_order=2, step_type="monitoring", depends_on="ensure_lakehouse_monitor")
pipeline.end_step(step, status="COMPLETED", output_summary=results.get("operational_metrics", {}))

step = pipeline.start_step("guardrail_stats", step_order=3, step_type="monitoring", depends_on="collect_metrics")
pipeline.end_step(step, status="COMPLETED", output_summary=results.get("guardrail_metrics", {}))

step = pipeline.start_step("drift_detection", step_order=4, step_type="monitoring", depends_on="guardrail_stats")
pipeline.end_step(step, status="COMPLETED", output_summary=results.get("drift_check", {}))

pipeline.end(status="COMPLETED")

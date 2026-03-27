# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse Monitoring Setup
# MAGIC Creates a monitor on the inference table (auto-captured by model serving).
# MAGIC Generates auto-refreshing dashboards for:
# MAGIC - Request/response data drift
# MAGIC - Token usage trends
# MAGIC - Latency distribution
# MAGIC - Error rates
# MAGIC - Guardrail block rates

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("chatbot_name", "agentops-docs-chatbot")
dbutils.widgets.text("monitoring_granularity", "1 day")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
chatbot_name = dbutils.widgets.get("chatbot_name")
granularity = dbutils.widgets.get("monitoring_granularity")

# Inference table name follows the pattern: {endpoint_name}_payload
inference_table = f"{catalog}.{schema}.`{chatbot_name}_payload`"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify inference table exists

# COMMAND ----------

try:
    df = spark.table(inference_table)
    count = df.count()
    print(f"Inference table: {inference_table}")
    print(f"  Records: {count}")
    print(f"  Columns: {', '.join(df.columns)}")
except Exception as e:
    print(f"Inference table not found: {inference_table}")
    print(f"The serving endpoint must have inference tables enabled and have received at least one request.")
    dbutils.notebook.exit('{"status": "SKIPPED", "reason": "inference table not found"}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Lakehouse Monitor

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorTimeSeries

w = WorkspaceClient()

monitor_name = f"{catalog}.{schema}.{chatbot_name}_monitor"

# Check if monitor already exists
try:
    existing = w.quality_monitors.get(table_name=inference_table)
    print(f"Monitor already exists: {monitor_name}")
    print(f"  Status: {existing.status}")
    print(f"  Dashboard: {existing.dashboard_id}")
except Exception:
    print(f"Creating monitor on {inference_table}...")

    # Create inference log monitor (specialized for serving endpoints)
    try:
        monitor = w.quality_monitors.create(
            table_name=inference_table,
            inference_log=MonitorInferenceLog(
                problem_type="text",
                prediction_col="response",
                timestamp_col="timestamp_ms",
                granularities=[granularity],
            ),
            output_schema_name=f"{catalog}.{schema}",
        )
        print(f"Monitor created!")
        print(f"  Dashboard ID: {monitor.dashboard_id}")
        print(f"  Status: {monitor.status}")
    except Exception as e:
        # Fall back to time series monitor if inference log doesn't work
        print(f"Inference log monitor failed ({e}), trying time series...")
        monitor = w.quality_monitors.create(
            table_name=inference_table,
            time_series=MonitorTimeSeries(
                timestamp_col="timestamp_ms",
                granularities=[granularity],
            ),
            output_schema_name=f"{catalog}.{schema}",
        )
        print(f"Time series monitor created!")
        print(f"  Dashboard ID: {monitor.dashboard_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run initial refresh

# COMMAND ----------

try:
    w.quality_monitors.run_refresh(table_name=inference_table)
    print("Monitor refresh triggered — dashboard will populate shortly")
except Exception as e:
    print(f"Refresh skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Show monitor details

# COMMAND ----------

monitor_info = w.quality_monitors.get(table_name=inference_table)

print(f"\n=== Lakehouse Monitor ===")
print(f"  Table: {inference_table}")
print(f"  Status: {monitor_info.status}")
print(f"  Granularity: {granularity}")
if hasattr(monitor_info, "dashboard_id") and monitor_info.dashboard_id:
    workspace_url = w.config.host
    print(f"  Dashboard: {workspace_url}/sql/dashboards/{monitor_info.dashboard_id}")
print(f"\n  Monitor output tables:")
print(f"    Profile: {catalog}.{schema}.{chatbot_name}_payload_profile_metrics")
print(f"    Drift:   {catalog}.{schema}.{chatbot_name}_payload_drift_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log to audit

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="setup_monitoring", agent_name=chatbot_name, environment="dev",
    triggered_by="manual",
)
pipeline.start()

step = pipeline.start_step("create_lakehouse_monitor", step_order=1, step_type="monitoring")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "inference_table": inference_table,
    "granularity": granularity,
    "dashboard_id": getattr(monitor_info, "dashboard_id", ""),
})

pipeline.end(status="COMPLETED")

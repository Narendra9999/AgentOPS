# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse Monitoring Setup
# MAGIC Creates a monitor on the inference table (auto-captured by model serving).
# MAGIC
# MAGIC **How it works:**
# MAGIC 1. Creates a parsed view on top of the raw payload table — extracts latency,
# MAGIC    status, user query, response length, and served entity from the JSON blobs
# MAGIC 2. Creates an InferenceLog monitor on the parsed view (falls back to TimeSeries)
# MAGIC 3. Triggers initial refresh → auto-generates a dashboard with:
# MAGIC    - Latency distribution (p50/p95/p99)
# MAGIC    - Error rate trends
# MAGIC    - Response length distribution
# MAGIC    - Drift detection across time windows
# MAGIC    - Per-model (champion/challenger) breakdowns

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("chatbot_name", "agentops-docs-chatbot")
dbutils.widgets.text("monitoring_granularity", "1 day")
dbutils.widgets.text("inference_table_name", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
chatbot_name = dbutils.widgets.get("chatbot_name")
granularity = dbutils.widgets.get("monitoring_granularity")
inference_table_override = dbutils.widgets.get("inference_table_name").strip()

# Use explicit inference table name if provided, otherwise derive from chatbot_name.
# Databricks auto-generates payload table names that may not match the chatbot_name
# (e.g. databricks_docs_agent_14_payload vs agentops-docs-chatbot_payload).
if inference_table_override:
    if "." in inference_table_override:
        inference_table = inference_table_override
    else:
        inference_table = f"{catalog}.{schema}.`{inference_table_override}`"
else:
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
# MAGIC ## 2. Create parsed view
# MAGIC Lakehouse Monitoring works best on structured numeric/categorical columns.
# MAGIC The raw payload table stores request/response as JSON strings — we extract
# MAGIC the useful fields into a view that the monitor can profile and track drift on.

# COMMAND ----------

# Derive view name from inference table name
_table_short = inference_table.split(".")[-1].strip("`")
parsed_view = f"{catalog}.{schema}.`{_table_short}_parsed`"

spark.sql(f"""
CREATE OR REPLACE VIEW {parsed_view} AS
SELECT
    request_time,
    request_date,
    status_code,
    execution_duration_ms AS latency_ms,
    served_entity_id AS model_id,

    -- Extract user query from request JSON
    get_json_object(request, '$.messages[0].content') AS user_query,
    length(get_json_object(request, '$.messages[0].content')) AS query_length,

    -- Extract assistant response from response JSON
    get_json_object(response, '$.messages[0].content') AS assistant_response,
    length(get_json_object(response, '$.messages[0].content')) AS response_length,

    -- Error indicator
    CASE WHEN status_code = 200 THEN 0 ELSE 1 END AS is_error,

    -- Requester for per-user analysis
    requester

FROM {inference_table}
""")

parsed_count = spark.table(parsed_view).count()
print(f"Parsed view created: {parsed_view}")
print(f"  Records: {parsed_count}")
print(f"  Columns: request_time, status_code, latency_ms, model_id, user_query,")
print(f"           query_length, assistant_response, response_length, is_error, requester")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Lakehouse Monitor
# MAGIC Uses InferenceLog type with the parsed view columns.

# COMMAND ----------

import requests as _req
import time

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_host = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _host.startswith("http"):
    _host = f"https://{_host}"
_headers = {"Authorization": f"Bearer {_token}", "Content-Type": "application/json"}

# Delete existing monitor if present (to allow re-creation with correct config)
_r = _req.get(f"{_host}/api/2.1/unity-catalog/tables/{parsed_view.replace('`', '')}/monitor", headers=_headers)
if _r.ok:
    print(f"Monitor already exists on {parsed_view} — deleting to recreate...")
    _req.delete(f"{_host}/api/2.1/unity-catalog/tables/{parsed_view.replace('`', '')}/monitor", headers=_headers)
    time.sleep(10)

print(f"Creating monitor on {parsed_view}...")

# Try InferenceLog monitor first, fall back to TimeSeries
_nb_context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
_user = _nb_context.userName().get()
_assets_dir = f"/Workspace/Users/{_user}/databricks_lakehouse_monitoring/{_table_short}_parsed"

_body = {
    "output_schema_name": f"{catalog}.{schema}",
    "assets_dir": _assets_dir,
    "inference_log": {
        "problem_type": "PROBLEM_TYPE_TEXT",
        "prediction_col": "assistant_response",
        "timestamp_col": "request_time",
        "model_id_col": "model_id",
        "granularities": [granularity],
    },
}
_r = _req.post(
    f"{_host}/api/2.1/unity-catalog/tables/{parsed_view.replace('`', '')}/monitor",
    headers=_headers, json=_body,
)
if _r.ok:
    monitor_type = "InferenceLog"
    print(f"InferenceLog monitor created!")
else:
    print(f"InferenceLog failed ({_r.status_code}: {_r.json().get('message', '')[:100]})")
    print(f"Falling back to TimeSeries...")
    _body = {
        "output_schema_name": f"{catalog}.{schema}",
        "assets_dir": _assets_dir,
        "time_series": {
            "timestamp_col": "request_time",
            "granularities": [granularity],
        },
    }
    _r = _req.post(
        f"{_host}/api/2.1/unity-catalog/tables/{parsed_view.replace('`', '')}/monitor",
        headers=_headers, json=_body,
    )
    if _r.ok:
        monitor_type = "TimeSeries"
        print(f"TimeSeries monitor created!")
    else:
        raise RuntimeError(f"Failed to create monitor: {_r.status_code} — {_r.text[:200]}")

monitor_data = _r.json()
dashboard_id = monitor_data.get("dashboard_id", "")
monitor_status = monitor_data.get("status", "")

print(f"  Type: {monitor_type}")
print(f"  Dashboard ID: {dashboard_id}")
print(f"  Status: {monitor_status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run initial refresh

# COMMAND ----------

_r = _req.post(
    f"{_host}/api/2.1/unity-catalog/tables/{parsed_view.replace('`', '')}/monitor/refreshes",
    headers=_headers,
)
if _r.ok:
    print("Monitor refresh triggered — dashboard will populate shortly")
else:
    print(f"Refresh skipped: {_r.status_code} — {_r.json().get('message', '')[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Show monitor details

# COMMAND ----------

_table_parsed_short = _table_short + "_parsed"

print(f"\n=== Lakehouse Monitor ===")
print(f"  Source table: {inference_table}")
print(f"  Parsed view:  {parsed_view}")
print(f"  Monitor type: {monitor_type}")
print(f"  Status: {monitor_status}")
print(f"  Granularity: {granularity}")
if dashboard_id:
    print(f"  Dashboard: {_host}/sql/dashboards/{dashboard_id}")
print(f"\n  Output tables:")
print(f"    Profile: {catalog}.{schema}.`{_table_parsed_short}_profile_metrics`")
print(f"    Drift:   {catalog}.{schema}.`{_table_parsed_short}_drift_metrics`")

print(f"\n  Monitored columns:")
print(f"    Numeric:     latency_ms, query_length, response_length, is_error, status_code")
print(f"    Categorical: model_id, requester")
print(f"    Text:        user_query, assistant_response")
print(f"    Timestamp:   request_time")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log to audit

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
    "parsed_view": parsed_view,
    "monitor_type": monitor_type,
    "granularity": granularity,
    "dashboard_id": dashboard_id,
})

pipeline.end(status="COMPLETED")

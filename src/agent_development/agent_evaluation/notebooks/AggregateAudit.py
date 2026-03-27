# Databricks notebook source
# MAGIC %md
# MAGIC # Audit Aggregation
# MAGIC Aggregates daily guardrail metrics into the audit summary table.
# MAGIC Runs on schedule (daily) via the audit_aggregation_job.

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("agent_name", "databricks_docs_agent")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
agent_name = dbutils.widgets.get("agent_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Collect guardrail metrics from traces (last 24h)

# COMMAND ----------

from framework.monitoring.trace_monitor import TraceMonitor

monitor = TraceMonitor(agent_name=agent_name, environment="dev")
metrics = monitor.collect_guardrail_metrics(hours=24)

print(f"Guardrail metrics (24h): {metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Write to guardrail_audit_log summary

# COMMAND ----------

import uuid
from datetime import datetime, timezone

# Ensure audit tables exist
from framework.audit.audit_logger import get_audit_ddls
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{audit_schema}")
for _tname, _ddl in get_audit_ddls(catalog, audit_schema).items():
    spark.sql(_ddl)

if metrics.get("total_requests", 0) > 0:
    summary_id = str(uuid.uuid4())
    block_reasons = str(metrics.get("block_reasons", {})).replace("'", "''")

    spark.sql(f"""
        INSERT INTO {catalog}.{audit_schema}.guardrail_audit_log VALUES (
            '{summary_id}', current_timestamp(),
            '{agent_name}', 'dev',
            'summary', 'daily_aggregate',
            false,
            'Daily: {metrics.get("total_requests", 0)} requests, {metrics.get("pre_llm_blocks", 0)} pre-blocks, {metrics.get("post_llm_blocks", 0)} post-blocks',
            'block_reasons: {block_reasons}'
        )""")
    print(f"Audit summary logged: {summary_id}")
else:
    print("No requests in the last 24 hours — nothing to aggregate")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Show recent audit entries

# COMMAND ----------

display(spark.sql(f"""
    SELECT * FROM {catalog}.{audit_schema}.guardrail_audit_log
    ORDER BY timestamp DESC
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log to audit

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="audit_aggregation", agent_name=agent_name, environment="dev",
    triggered_by="schedule", depends_on="none", spark=spark,
)
pipeline.start()

step = pipeline.start_step("aggregate_guardrail_metrics", step_order=1, step_type="audit", depends_on="none")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "total_requests": metrics.get("total_requests", 0),
    "pre_llm_blocks": metrics.get("pre_llm_blocks", 0),
    "post_llm_blocks": metrics.get("post_llm_blocks", 0),
    "block_reasons": metrics.get("block_reasons", {}),
})

pipeline.end(status="COMPLETED")

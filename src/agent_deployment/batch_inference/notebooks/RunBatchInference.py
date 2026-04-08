# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference Pipeline
# MAGIC Process a table of input queries through the agent endpoint.
# MAGIC Guardrail-blocked records go to a quarantine table for review.

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("batch_input_table", "batch_input")
dbutils.widgets.text("batch_output_table", "batch_output")
dbutils.widgets.text("batch_quarantine_table", "batch_quarantine")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
agent_name = dbutils.widgets.get("agent_name")
input_table = dbutils.widgets.get("batch_input_table")
output_table = dbutils.widgets.get("batch_output_table")
quarantine_table = dbutils.widgets.get("batch_quarantine_table")

# Find the deployed endpoint by prefix (handles truncated names from agents.deploy)
from databricks.sdk import WorkspaceClient
_w = WorkspaceClient()
_match_prefix = f"agents_{catalog}-{schema}"
chatbot_name = None
for _ep in _w.serving_endpoints.list():
    if _ep.name.startswith(_match_prefix) and _ep.state and str(_ep.state.ready).endswith("READY"):
        chatbot_name = _ep.name
        break
if not chatbot_name:
    dbutils.notebook.exit('{"status": "SKIPPED", "reason": "no READY endpoint found"}')
print(f"Using endpoint: {chatbot_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preview input table

# COMMAND ----------

fq_input = f"{catalog}.{schema}.{input_table}"
try:
    input_df = spark.table(fq_input)
    print(f"Input records: {input_df.count()} from {fq_input}")
    display(input_df.limit(5))
except Exception as e:
    print(f"Input table not found: {fq_input}")
    print(f"Create it first with columns: question (STRING), plus any metadata columns")
    print(f"Example:")
    print(f"  CREATE TABLE {fq_input} (question STRING, source STRING, priority STRING)")
    dbutils.notebook.exit('{"status": "SKIPPED", "reason": "no input table"}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run batch inference

# COMMAND ----------

from framework.inference.batch_runner import BatchInferenceRunner

runner = BatchInferenceRunner(
    endpoint_name=chatbot_name,
    catalog=catalog,
    schema=schema,
)

result = runner.run_batch(
    input_table=input_table,
    output_table=output_table,
    quarantine_table=quarantine_table,
    query_column="question",
)

print(f"\n=== Batch Inference Results ===")
print(f"  Status:      {result['status']}")
print(f"  Input:       {result.get('input_records', 0)} records")
print(f"  Output:      {result.get('output_records', 0)} records")
print(f"  Quarantined: {result.get('quarantined_records', 0)} records")
print(f"  Duration:    {result.get('duration_seconds', 0)}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preview results

# COMMAND ----------

if result["status"] == "COMPLETED" and result.get("output_records", 0) > 0:
    print("=== Output (sample) ===")
    display(spark.table(f"{catalog}.{schema}.{output_table}").limit(5))

# COMMAND ----------

if result["status"] == "COMPLETED" and result.get("quarantined_records", 0) > 0:
    print("=== Quarantined Records (guardrail-blocked) ===")
    display(spark.table(f"{catalog}.{schema}.{quarantine_table}"))
else:
    print("No quarantined records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log to audit

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger
import json

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="batch_inference", agent_name=agent_name, environment="dev",
    triggered_by="schedule", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("run_batch", step_order=1, step_type="inference")
pipeline.end_step(
    step,
    status=result["status"],
    records_processed=result.get("output_records", 0),
    output_summary={
        "input_records": result.get("input_records", 0),
        "output_records": result.get("output_records", 0),
        "quarantined_records": result.get("quarantined_records", 0),
    },
    error_message=result.get("error", ""),
)

pipeline.end(status=result["status"])

# COMMAND ----------

dbutils.notebook.exit(json.dumps(result))

# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search Setup
# MAGIC Create endpoint and delta sync index for RAG retrieval.

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("preprocessed_data_table", "databricks_docs_chunked")
dbutils.widgets.text("vs_endpoint", "agentops-vs-endpoint")
dbutils.widgets.text("vs_index", "databricks_docs_index")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
preprocessed_data_table = dbutils.widgets.get("preprocessed_data_table")
vs_endpoint = dbutils.widgets.get("vs_endpoint")
vs_index_name = dbutils.widgets.get("vs_index")
embedding_model = dbutils.widgets.get("embedding_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Start audit tracking

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="vector_search_setup", agent_name="", environment="dev",
    triggered_by="pipeline", depends_on="data_preprocessing", spark=spark, dbutils=dbutils,
)
pipeline.start()
step = pipeline.start_step("create_index", step_order=1, step_type="data_prep", depends_on="data_preprocessing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create vector search endpoint

# COMMAND ----------

import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)  # .../notebooks
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_dir)))
sys.path.insert(0, _project_root)
print(f"Project root: {_project_root}")

from data_preparation.vector_search.vector_search_utils.utils import (
    get_or_create_endpoint, create_delta_sync_index)

endpoint = get_or_create_endpoint(vs_endpoint)
print(f"Endpoint ready: {vs_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create delta sync index

# COMMAND ----------

index_full_name = f"{catalog}.{schema}.{vs_index_name}"
source_table = f"{catalog}.{schema}.{preprocessed_data_table}"

index = create_delta_sync_index(
    endpoint_name=vs_endpoint,
    index_name=index_full_name,
    source_table=source_table,
    embedding_model=embedding_model,
    text_column="chunk_text",
    primary_key="chunk_id",
)

print(f"Index ready: {index_full_name}")
print(f"  Source table: {source_table}")
print(f"  Endpoint: {vs_endpoint}")
print(f"  Embedding model: {embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test query

# COMMAND ----------

import time
print("Waiting 30s for initial sync...")
time.sleep(30)

from data_preparation.vector_search.vector_search_utils.utils import query_index

try:
    results = query_index(
        index_name=index_full_name,
        query_text="How do I create a Unity Catalog table?",
        num_results=3,
    )
    if results and results.get("result", {}).get("data_array"):
        for row in results["result"]["data_array"]:
            print(f"  [{row[2]}] {row[0][:100]}...")
    else:
        print("No results yet — index may still be syncing.")
except Exception as e:
    print(f"Query test skipped (index still syncing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Complete audit tracking

# COMMAND ----------

pipeline.end_step(step, status="COMPLETED", output_summary={
    "vs_endpoint": vs_endpoint,
    "vs_index": index_full_name,
    "source_table": source_table,
    "embedding_model": embedding_model,
})
pipeline.end(status="COMPLETED")

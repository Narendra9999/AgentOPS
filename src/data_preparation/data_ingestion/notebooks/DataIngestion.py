# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion
# MAGIC Scrape Databricks documentation from the sitemap and store in Unity Catalog.

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("data_source_url", "https://docs.databricks.com/en/doc-sitemap.xml")
dbutils.widgets.text("max_documents", "0")
dbutils.widgets.text("raw_data_table", "databricks_docs_raw")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
data_source_url = dbutils.widgets.get("data_source_url")
max_docs = int(dbutils.widgets.get("max_documents")) or None
raw_data_table = dbutils.widgets.get("raw_data_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Set catalog and create schema

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
print(f"Schema ready: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Install dependencies and fetch docs

# COMMAND ----------

# Install dependencies — uses air-gapped volume if available, otherwise PyPI
import subprocess, os
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
if os.path.exists(_vol_path):
    subprocess.check_call(["pip", "install", "beautifulsoup4", "lxml", "requests", "--find-links", _vol_path, "--no-index", "-q"])
else:
    subprocess.check_call(["pip", "install", "beautifulsoup4", "lxml", "requests", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
data_source_url = dbutils.widgets.get("data_source_url")
max_docs = int(dbutils.widgets.get("max_documents")) or None
raw_data_table = dbutils.widgets.get("raw_data_table")

# Start audit tracking (after pip restart so variables persist)
from framework.audit.audit_logger import PipelineStepLogger
pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="data_ingestion", agent_name="", environment="dev",
    triggered_by="pipeline", depends_on="none", spark=spark,
)
pipeline.start()
step = pipeline.start_step("fetch_and_save", step_order=1, step_type="data_prep", depends_on="none")

# COMMAND ----------

# Add project to Python path for imports
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)  # .../notebooks
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_dir)))
sys.path.insert(0, _project_root)
print(f"Project root: {_project_root}")

from data_preparation.data_ingestion.ingestion.fetch_data import fetch_data_from_url, load_data_from_file

# Check if data_source_url is a local file or a URL
if data_source_url.startswith("http"):
    print(f"Fetching docs from URL: {data_source_url}")
    if max_docs:
        print(f"Limited to: {max_docs} documents")
    docs_df = fetch_data_from_url(spark, data_source_url, max_documents=max_docs)
else:
    # Local file path — bundled dataset for air-gapped environments
    # _project_root = .../files/src, bundle root = .../files (where fixtures/ lives)
    _bundle_root = os.path.dirname(_project_root)
    _local_path = os.path.join(_bundle_root, "fixtures", "databricks_docs.json")
    print(f"Loading docs from local dataset: {_local_path}")
    docs_df = load_data_from_file(spark, _local_path, max_documents=max_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save raw docs to Unity Catalog

# COMMAND ----------

table_name = f"{catalog}.{schema}.{raw_data_table}"
docs_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

count = spark.table(table_name).count()
print(f"Saved {count} documents to {table_name}")

# COMMAND ----------

display(spark.table(table_name).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Complete audit tracking

# COMMAND ----------

pipeline.end_step(step, status="COMPLETED", records_processed=count, output_summary={
    "data_source_url": data_source_url,
    "raw_data_table": table_name,
    "document_count": count,
})
pipeline.end(status="COMPLETED")

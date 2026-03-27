# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preprocessing
# MAGIC Clean HTML, chunk documents, and prepare for vector search.

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("raw_data_table", "databricks_docs_raw")
dbutils.widgets.text("preprocessed_data_table", "databricks_docs_chunked")
dbutils.widgets.text("chunk_size", "1000")
dbutils.widgets.text("chunk_overlap", "200")
dbutils.widgets.text("min_chunk_size", "50")
dbutils.widgets.text("chunking_strategy", "sentence")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_data_table = dbutils.widgets.get("raw_data_table")
preprocessed_data_table = dbutils.widgets.get("preprocessed_data_table")
chunk_size = int(dbutils.widgets.get("chunk_size"))
chunk_overlap = int(dbutils.widgets.get("chunk_overlap"))
min_chunk_size = int(dbutils.widgets.get("min_chunk_size"))
chunking_strategy = dbutils.widgets.get("chunking_strategy")

print(f"Config: strategy={chunking_strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}, min={min_chunk_size}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Start audit tracking

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="data_preprocessing", agent_name="", environment="dev",
    triggered_by="pipeline", depends_on="data_ingestion", spark=spark,
)
pipeline.start()
step = pipeline.start_step("chunk_documents", step_order=1, step_type="data_prep", depends_on="data_ingestion")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read raw docs

# COMMAND ----------

raw_table = f"{catalog}.{schema}.{raw_data_table}"
raw_df = spark.table(raw_table)
print(f"Raw documents: {raw_df.count()} from {raw_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clean HTML and chunk

# COMMAND ----------

from pyspark.sql.functions import col, explode, concat_ws
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
from pyspark.sql import functions as F
import pandas as pd
import sys, os
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)  # .../notebooks
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_dir)))
sys.path.insert(0, _project_root)
print(f"Project root: {_project_root}")

from data_preparation.data_preprocessing.preprocessing.create_chunk import (
    clean_html, chunk_text)

# Capture variables for UDF closure
_chunk_size = chunk_size
_chunk_overlap = chunk_overlap
_min_chunk_size = min_chunk_size
_chunking_strategy = chunking_strategy

chunk_schema = ArrayType(StructType([
    StructField("chunk_text", StringType()),
    StructField("chunk_index", IntegerType()),
]))

@F.pandas_udf(chunk_schema)
def chunk_udf(texts: pd.Series) -> pd.Series:
    results = []
    for text in texts:
        clean = clean_html(text) if text else ""
        chunks = chunk_text(clean, _chunk_size, _chunk_overlap, _chunking_strategy)
        # Filter out chunks shorter than min_chunk_size
        results.append([
            {"chunk_text": c, "chunk_index": i}
            for i, c in enumerate(chunks)
            if len(c) >= _min_chunk_size
        ])
    return pd.Series(results)

# COMMAND ----------

chunked_df = (
    raw_df
    .withColumn("chunks", chunk_udf(col("text")))
    .select("url", explode("chunks").alias("chunk"))
    .select(
        col("url"),
        col("chunk.chunk_text").alias("chunk_text"),
        col("chunk.chunk_index").alias("chunk_index"),
    )
    .withColumn("chunk_id", concat_ws("_", col("url"), col("chunk_index")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save chunked docs (with change data feed for vector search sync)

# COMMAND ----------

table_name = f"{catalog}.{schema}.{preprocessed_data_table}"
(
    chunked_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(table_name)
)

count = spark.table(table_name).count()
print(f"Created {count} chunks in {table_name}")
print(f"  chunk_size={chunk_size}, overlap={chunk_overlap}, min={min_chunk_size}")

# COMMAND ----------

display(spark.table(table_name).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Complete audit tracking

# COMMAND ----------

pipeline.end_step(step, status="COMPLETED", records_processed=count, output_summary={
    "preprocessed_table": table_name,
    "chunk_count": count,
    "chunking_strategy": chunking_strategy,
    "chunk_size": chunk_size,
    "chunk_overlap": chunk_overlap,
})
pipeline.end(status="COMPLETED")

"""
Export fresh databricks docs from Unity Catalog table to fixtures/databricks_docs.json.

Run this AFTER the pipeline completes on a workspace with internet access (e.g. dev).
The exported JSON can then be used for air-gapped deployments (e.g. Mastercard).

Usage (from notebook):
    %run ../scripts/export_docs_fixture

Usage (from CLI):
    databricks workspace export ... (then run locally)
"""

import json
import os

# Config — adjust if needed
CATALOG = "classic_stable_cykcbe_catalog"
SCHEMA = "agentops"
RAW_TABLE = "databricks_docs_raw"

# Read from UC table
table_name = f"{CATALOG}.{SCHEMA}.{RAW_TABLE}"
print(f"Reading from {table_name}...")
df = spark.table(table_name).select("url", "text")

# Convert to JSON
docs = [row.asDict() for row in df.collect()]
print(f"Total docs: {len(docs)}")

# Check content quality
truncated = sum(1 for d in docs if len(d.get("text", "")) == 5000)
avg_len = sum(len(d.get("text", "")) for d in docs) / len(docs) if docs else 0
max_len = max(len(d.get("text", "")) for d in docs) if docs else 0
print(f"Avg doc length: {avg_len:.0f} chars")
print(f"Max doc length: {max_len} chars")
print(f"Docs at exactly 5000 chars: {truncated} (should be 0 after re-scrape)")

# Save to fixtures
# Note: when running in notebook, _project_root points to .../files/src
# fixtures/ is at .../files/fixtures/
import sys
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(_nb_dir))
_fixture_path = os.path.join(_project_root, "fixtures", "databricks_docs.json")

print(f"Saving to {_fixture_path}...")
with open(_fixture_path, "w") as f:
    json.dump(docs, f, indent=2, ensure_ascii=False)

file_size = os.path.getsize(_fixture_path) / (1024 * 1024)
print(f"Done! File size: {file_size:.1f} MB")
print(f"Copy this file to your local project: fixtures/databricks_docs.json")

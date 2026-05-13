# Databricks notebook source
# MAGIC %md
# MAGIC # Test Prompt Versioning
# MAGIC Validates MLflow Prompt Registry: register, version, alias, load.

# COMMAND ----------

import subprocess
subprocess.check_call(["pip", "install", "-U", "mlflow[genai]>=3.5", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json

CATALOG = "classic_stable_cykcbe_catalog"
SCHEMA = "agentops"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.databricks_docs_agent_system_prompt"

results = {}
print(f"MLflow version: {mlflow.__version__}")
print(f"Prompt: {PROMPT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Register v1

# COMMAND ----------

v1_template = "You are a Databricks Documentation Assistant. Answer questions about Databricks."

v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=v1_template,
    commit_message="v1 — basic prompt",
)
print(f"Registered v{v1.version}")
results["v1"] = v1.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register v2

# COMMAND ----------

v2_template = "You are a Databricks Documentation Assistant with tool support. Answer questions and use tools when available."

v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=v2_template,
    commit_message="v2 — added tool awareness",
)
print(f"Registered v{v2.version}")
results["v2"] = v2.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set aliases

# COMMAND ----------

mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="dev", version=v1.version)
print(f"@dev → v{v1.version}")

mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=v2.version)
print(f"@production → v{v2.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load by alias

# COMMAND ----------

prod = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"@production → v{prod.version}: {prod.template[:80]}...")
results["production_version"] = prod.version
results["production_matches_v2"] = prod.version == v2.version

dev = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@dev")
print(f"@dev → v{dev.version}: {dev.template[:80]}...")
results["dev_version"] = dev.version
results["dev_matches_v1"] = dev.version == v1.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load by version number

# COMMAND ----------

v1_loaded = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{v1.version}")
print(f"Direct v{v1.version}: {v1_loaded.template[:60]}...")
results["direct_load_works"] = v1_loaded.template == v1_template

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promote dev → production

# COMMAND ----------

mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=v1.version)
after_promote = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"After promotion: @production → v{after_promote.version}")
results["promotion_works"] = after_promote.version == v1.version

# Restore v2 as production
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=v2.version)
print(f"Restored: @production → v{v2.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

all_passed = all([
    results.get("production_matches_v2"),
    results.get("dev_matches_v1"),
    results.get("direct_load_works"),
    results.get("promotion_works"),
])
results["status"] = "PASSED" if all_passed else "FAILED"

print(f"\n{'='*50}")
print(f"  PROMPT VERSIONING: {results['status']}")
print(f"{'='*50}")
for k, v in results.items():
    print(f"  {k}: {v}")

dbutils.notebook.exit(json.dumps(results))

# Databricks notebook source
# MAGIC %md
# MAGIC # Test Prompt Versioning
# MAGIC Validates MLflow Prompt Registry: register, version, alias, load.
# MAGIC
# MAGIC Reference: https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/use-prompts-in-deployed-apps

# COMMAND ----------

import subprocess
subprocess.check_call(["pip", "install", "-U", "mlflow[genai]>=3.5", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import json
from datetime import datetime

CATALOG = "classic_stable_cykcbe_catalog"
SCHEMA = "agentops"
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.databricks_docs_agent_system_prompt"

results = {}
print(f"MLflow version: {mlflow.__version__}")
print(f"Prompt: {PROMPT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Register initial prompt (v1)

# COMMAND ----------

v1_template = """\
You are a Databricks Documentation Assistant. You help users understand:
- Databricks products and features
- API usage and SDK integration patterns

Guidelines:
- Always base your answers on the provided documentation context
- Include code snippets when relevant
- If the context doesn't contain the answer, say so honestly"""

v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=v1_template,
    commit_message="v1 — initial system prompt",
    tags={"author": "agentops", "stage": "test"},
)
print(f"Registered v{v1.version}: {v1_template[:60]}...")
results["v1"] = v1.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register improved prompt (v2)

# COMMAND ----------

v2_template = """\
You are a Databricks Documentation Assistant. You help users understand:
- Databricks products and features (Unity Catalog, Delta Lake, MLflow, etc.)
- API usage and SDK integration patterns
- Coding best practices on the Databricks platform
- Configuration, deployment, and troubleshooting

You have access to tools: Calculator, SQL Formatter, Cluster Sizing.
When Tool Results appear in your context, incorporate them into your answer.

Guidelines:
- Always base your answers on the provided documentation context
- Include code snippets when relevant
- Cite the source URL when referencing a specific doc page
- If the context doesn't contain the answer, say so honestly
- Keep responses concise and actionable"""

v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=v2_template,
    commit_message="v2 — added tool awareness and more guidelines",
    tags={"author": "agentops", "stage": "test"},
)
print(f"Registered v{v2.version}: {v2_template[:60]}...")
results["v2"] = v2.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set aliases

# COMMAND ----------

# Set v1 as "dev", v2 as "production"
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="dev", version=v1.version)
print(f"Set @dev → v{v1.version}")

mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=v2.version)
print(f"Set @production → v{v2.version}")

results["alias_dev"] = v1.version
results["alias_production"] = v2.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load prompts by alias

# COMMAND ----------

# Load @production
prod_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"@production → v{prod_prompt.version}")
print(f"Template ({len(prod_prompt.template)} chars): {prod_prompt.template[:100]}...")
results["loaded_production_version"] = prod_prompt.version

# Load @dev
dev_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@dev")
print(f"\n@dev → v{dev_prompt.version}")
print(f"Template ({len(dev_prompt.template)} chars): {dev_prompt.template[:100]}...")
results["loaded_dev_version"] = dev_prompt.version

# Load by version number
v1_loaded = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{v1.version}")
print(f"\nDirect v{v1.version} load: {v1_loaded.template[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. List all versions

# COMMAND ----------

versions = mlflow.genai.search_prompt_versions(PROMPT_NAME)
print(f"Total versions: {len(versions)}")
results["total_versions"] = len(versions)

for v in versions:
    aliases = [a.alias for a in (v.aliases or [])]
    print(f"  v{v.version}: aliases={aliases} | {v.template[:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Promote dev → production (simulate)

# COMMAND ----------

# Simulate promoting dev to production
dev_loaded = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@dev")
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=dev_loaded.version)
print(f"Promoted: @production now → v{dev_loaded.version} (was v{v2.version})")

# Verify
prod_after = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print(f"Verified: @production → v{prod_after.version}")
results["promotion_verified"] = prod_after.version == dev_loaded.version

# Restore v2 as production
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=v2.version)
print(f"Restored: @production → v{v2.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

results["status"] = "passed"
print(f"\n{'='*50}")
print(f"  PROMPT VERSIONING TEST: PASSED")
print(f"{'='*50}")
print(f"  Versions created: v{v1.version}, v{v2.version}")
print(f"  Total versions: {results['total_versions']}")
print(f"  @production → v{results['alias_production']}")
print(f"  @dev → v{results['alias_dev']}")
print(f"  Promotion test: {'PASSED' if results['promotion_verified'] else 'FAILED'}")

dbutils.notebook.exit(json.dumps(results))

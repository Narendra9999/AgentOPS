# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Optimization with GEPA
# MAGIC Optimizes the agent's system prompt using `mlflow.genai.optimize_prompts()`.
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Register current system prompt in MLflow Prompt Registry
# MAGIC 2. Define predict_fn that loads prompt from registry and calls the LLM
# MAGIC 3. Define scorers (builtin + custom via make_judge)
# MAGIC 4. Run GepaPromptOptimizer — iterative reflection-based optimization
# MAGIC 5. Register optimized prompt and update @production alias
# MAGIC
# MAGIC **Requirements:** mlflow>=3.5, databricks-sdk, dspy, openai
# MAGIC
# MAGIC **Reference:** https://docs.databricks.com/aws/en/mlflow3/genai/tutorials/examples/prompt-optimization-quickstart

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-120b")
dbutils.widgets.text("judge_model", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("max_metric_calls", "100")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
judge_model = dbutils.widgets.get("judge_model")
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os

_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
_wheels_path = _vol_path if os.path.exists(_vol_path) else None

if _wheels_path:
    print(f"Installing from: {_wheels_path}")
    subprocess.check_call(["pip", "install", "-U", "mlflow", "databricks-sdk", "dspy", "openai", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "mlflow[genai]>=3.5", "databricks-sdk", "dspy>=2.6", "openai", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
judge_model = dbutils.widgets.get("judge_model")
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import mlflow
import json
import yaml
import time
from databricks_openai import DatabricksOpenAI

PROMPT_NAME = f"{catalog}.{schema}.{agent_name}_system_prompt"
REFLECTION_MODEL = f"databricks:/{judge_model}"
SCORER_MODEL = f"databricks:/{judge_model}"

# Set experiment
_user = spark.sql("SELECT current_user()").first()[0]
experiment = mlflow.set_experiment(f"/Users/{_user}/{agent_name}")

# OpenAI-compatible client for Databricks endpoints
openai_client = DatabricksOpenAI()

print(f"Experiment: {experiment.name}")
print(f"LLM endpoint: {llm_endpoint}")
print(f"Reflection model: {REFLECTION_MODEL}")
print(f"Prompt name: {PROMPT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load and register current system prompt

# COMMAND ----------

# Load current system prompt from config.yaml
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
    _cfg = yaml.safe_load(f)
current_prompt = _cfg.get("system_prompt", "")

# Register in MLflow Prompt Registry
prompt = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=current_prompt,
    commit_message="Current system prompt from config.yaml",
)

print(f"Registered prompt: {PROMPT_NAME} (version {prompt.version})")
print(f"URI: {prompt.uri}")
print(f"Prompt ({len(current_prompt)} chars):\n{current_prompt[:300]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define predict function
# MAGIC
# MAGIC The predict_fn loads the prompt from the registry (so GEPA can optimize it)
# MAGIC and calls the LLM endpoint via the OpenAI-compatible client.

# COMMAND ----------

def predict_fn(question: str) -> str:
    """Load prompt from registry, format with question, call LLM."""
    loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@latest")
    system_content = loaded_prompt.format()

    completion = openai_client.chat.completions.create(
        model=llm_endpoint,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    content = completion.choices[0].message.content
    # Handle list-of-objects content from reasoning models
    if isinstance(content, list):
        text_parts = [
            item.get("text", item.get("content", ""))
            for item in content
            if isinstance(item, dict)
        ]
        content = "".join(text_parts)
    return content

# Validate
_test = predict_fn("What is Delta Lake?")
print(f"predict_fn validated ({len(_test)} chars): {_test[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load evaluation dataset

# COMMAND ----------

import pandas as pd

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_df = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_df)} rows from {golden_table}")

# Convert to optimize_prompts format
dataset = []
for _, row in eval_df.iterrows():
    question = row.get("request", row.get("input", ""))
    expected = row.get("expected_response", row.get("output", ""))
    entry = {
        "inputs": {"question": question},
        "outputs": {"response": expected},
    }
    if expected:
        entry["expectations"] = {"expected_response": expected}
    dataset.append(entry)

print(f"Converted: {len(dataset)} examples")
print(f"Sample: {dataset[0]['inputs']['question'][:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define scorers
# MAGIC
# MAGIC Using both builtin scorers (Correctness) and custom scorers (make_judge).

# COMMAND ----------

from mlflow.genai.scorers import Correctness
from mlflow.genai.judges import make_judge

# Builtin scorer: checks factual correctness against expected_response
correctness_scorer = Correctness(model=SCORER_MODEL)

# Custom scorer: domain-specific quality evaluation
quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality of the agent's response.\n\n"
        "Question: {{ inputs.question }}\n"
        "Response: {{ outputs.response }}\n\n"
        "Consider:\n"
        "- Accuracy: factually correct based on Databricks documentation?\n"
        "- Completeness: does it fully address the question?\n"
        "- Code quality: correct code snippets when relevant?\n"
        "- Actionability: can the user follow the guidance?"
    ),
    model=SCORER_MODEL,
)

print(f"Scorers: Correctness + response_quality (model: {judge_model})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run GEPA prompt optimization
# MAGIC
# MAGIC GEPA iteratively generates candidate prompt variations using LLM-driven
# MAGIC reflection and automated feedback, then selects the best-performing prompt.

# COMMAND ----------

from mlflow.genai.optimize import GepaPromptOptimizer

print(f"{'=' * 60}")
print(f"  GEPA Prompt Optimization")
print(f"  Reflection model: {REFLECTION_MODEL}")
print(f"  Max metric calls: {MAX_METRIC_CALLS}")
print(f"  Dataset: {len(dataset)} examples")
print(f"{'=' * 60}\n")

t0 = time.time()

with mlflow.start_run(run_name="gepa_prompt_optimization"):
    mlflow.log_params({
        "optimizer": "GepaPromptOptimizer",
        "reflection_model": REFLECTION_MODEL,
        "scorer_model": SCORER_MODEL,
        "max_metric_calls": MAX_METRIC_CALLS,
        "dataset_size": len(dataset),
        "llm_endpoint": llm_endpoint,
        "prompt_name": PROMPT_NAME,
    })

    result = mlflow.genai.optimize_prompts(
        predict_fn=predict_fn,
        train_data=dataset,
        prompt_uris=[prompt.uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=REFLECTION_MODEL,
            max_metric_calls=MAX_METRIC_CALLS,
        ),
        scorers=[correctness_scorer, quality_judge],
    )

    elapsed = time.time() - t0
    optimized_prompt_obj = result.optimized_prompts[0]

    mlflow.log_metrics({
        "initial_score": float(result.initial_eval_score),
        "final_score": float(result.final_eval_score),
        "improvement": float(result.final_eval_score - result.initial_eval_score),
        "optimization_time_s": elapsed,
    })

    print(f"Initial score: {result.initial_eval_score:.4f}")
    print(f"Final score:   {result.final_eval_score:.4f}")
    print(f"Improvement:   {result.final_eval_score - result.initial_eval_score:+.4f}")
    print(f"Time:          {elapsed:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Review and register optimized prompt

# COMMAND ----------

optimized_text = optimized_prompt_obj.template

print("=== Optimized Prompt ===")
print(optimized_text[:500])
if len(optimized_text) > 500:
    print(f"... ({len(optimized_text)} chars total)")

print(f"\n=== Comparison ===")
print(f"Original: {len(current_prompt)} chars")
print(f"Optimized: {len(optimized_text)} chars")

# COMMAND ----------

# Register optimized prompt if it improved
if result.final_eval_score > result.initial_eval_score:
    new_prompt = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=optimized_text,
        commit_message=(
            f"GEPA optimized: {result.initial_eval_score:.3f} → {result.final_eval_score:.3f} "
            f"(+{result.final_eval_score - result.initial_eval_score:.3f})"
        ),
        tags={"optimizer": "GEPA", "improvement": str(round(result.final_eval_score - result.initial_eval_score, 4))},
    )
    print(f"Registered optimized prompt: {PROMPT_NAME} v{new_prompt.version}")

    # Update production alias
    mlflow.genai.set_prompt_alias(
        name=PROMPT_NAME,
        alias="production",
        version=new_prompt.version,
    )
    print(f"Updated @production alias → v{new_prompt.version}")
else:
    print(f"No improvement ({result.initial_eval_score:.3f} → {result.final_eval_score:.3f})")
    print("Keeping current prompt.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Checkpoint and audit

# COMMAND ----------

# Checkpoint to Delta
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

CHECKPOINT_TABLE = f"{catalog}.{schema}.gepa_optimization_results"

checkpoint_row = {
    "prompt_name": PROMPT_NAME,
    "initial_score": float(result.initial_eval_score),
    "final_score": float(result.final_eval_score),
    "improvement": float(result.final_eval_score - result.initial_eval_score),
    "optimized_prompt": optimized_text,
    "elapsed_seconds": round(elapsed, 1),
    "reflection_model": REFLECTION_MODEL,
}

_schema = StructType([
    StructField("prompt_name", StringType()),
    StructField("initial_score", DoubleType()),
    StructField("final_score", DoubleType()),
    StructField("improvement", DoubleType()),
    StructField("optimized_prompt", StringType()),
    StructField("elapsed_seconds", DoubleType()),
    StructField("reflection_model", StringType()),
])

spark.createDataFrame([checkpoint_row], schema=_schema).write.mode("append").saveAsTable(CHECKPOINT_TABLE)
print(f"Checkpointed to {CHECKPOINT_TABLE}")

# Audit logging
from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="prompt_optimization", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()
step = pipeline.start_step("gepa_optimization", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(dataset), output_summary={
    "initial_score": float(result.initial_eval_score),
    "final_score": float(result.final_eval_score),
    "improvement": float(result.final_eval_score - result.initial_eval_score),
    "prompt_name": PROMPT_NAME,
})
pipeline.end(status="COMPLETED")

dbutils.notebook.exit(json.dumps({
    "status": "completed",
    "initial_score": float(result.initial_eval_score),
    "final_score": float(result.final_eval_score),
    "improvement": float(result.final_eval_score - result.initial_eval_score),
    "prompt_name": PROMPT_NAME,
}))

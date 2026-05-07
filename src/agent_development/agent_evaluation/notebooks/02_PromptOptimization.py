# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Prompt Optimization with DSPy MIPROv2
# MAGIC Runs DSPy MIPROv2 prompt optimization using the aligned judge from notebook 05 as the scorer.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Aligned judge registered (from 05_JudgeAlignment)
# MAGIC - Evaluation dataset in UC table
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads the aligned judge from the experiment
# MAGIC 2. Loads the current system prompt from config / MLflow Prompt Registry
# MAGIC 3. Runs N DSPy MIPROv2 optimization rounds, checkpointing results to Delta
# MAGIC 4. Selects the best prompt and registers it in the MLflow Prompt Registry
# MAGIC
# MAGIC **Reference:** Notebook 06-PromptOptimization from at-bat-assistant

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("judge_model", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("aligned_judge_name", "response_quality_aligned")
dbutils.widgets.text("n_runs", "3")
dbutils.widgets.text("max_metric_calls", "100")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
N_RUNS = int(dbutils.widgets.get("n_runs"))
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os

# Find wheels: bundled (DAB) → Mastercard volume → FEVM volume → PyPI
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_eval_dir = "/Workspace" + os.path.dirname(os.path.dirname(_nb_path))
_wheels_path = None
for _candidate in [
    os.path.join(_eval_dir, "wheels"),
    "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs",
    "/Volumes/classic_stable_cykcbe_catalog/agentops/app_wheels",
]:
    if os.path.exists(_candidate) and any(f.endswith(".whl") for f in os.listdir(_candidate)):
        _wheels_path = _candidate
        break

if _wheels_path:
    print(f"Installing from: {_wheels_path}")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "dspy", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.4,<3.11", "dspy>=2.6", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
N_RUNS = int(dbutils.widgets.get("n_runs"))
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure environment

# COMMAND ----------

import os, json, yaml, warnings, logging
import mlflow

# Suppress noisy logs during optimization
warnings.filterwarnings("ignore")
logging.getLogger("mlflow.genai.judges.instructions_judge").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracing.fluent").setLevel(logging.ERROR)

# Databricks FMAPI routing
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _ws_url.startswith("http"):
    _ws_url = f"https://{_ws_url}"
os.environ["DATABRICKS_API_KEY"] = _token
os.environ["DATABRICKS_API_BASE"] = f"{_ws_url}/serving-endpoints"
os.environ["DATABRICKS_HOST"] = _ws_url
os.environ["DATABRICKS_TOKEN"] = _token

REFLECTION_MODEL = f"databricks:/{judge_model}"
PROMPT_NAME = f"{catalog}.{schema}.{agent_name}_system_prompt"
CHECKPOINT_TABLE = f"{catalog}.{schema}.dspy_mipro_experiment_checkpoint"

# Set experiment
_user = spark.sql("SELECT current_user()").first()[0]
experiment = mlflow.set_experiment(f"/Users/{_user}/{agent_name}")
EXPERIMENT_ID = experiment.experiment_id

print(f"Experiment: {experiment.name} (ID: {EXPERIMENT_ID})")
print(f"Reflection model: {REFLECTION_MODEL}")
print(f"Prompt name: {PROMPT_NAME}")
print(f"Checkpoint table: {CHECKPOINT_TABLE}")
print(f"Optimization: {N_RUNS} runs, max {MAX_METRIC_CALLS} metric calls each")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load aligned judge

# COMMAND ----------

from mlflow.genai.scorers import get_scorer

# Try loading persisted aligned judge; fall back to recreating alignment
try:
    aligned_judge = get_scorer(name=aligned_judge_name)
    print(f"Loaded aligned judge from registry: {aligned_judge.name}")
except Exception as e:
    print(f"Could not load aligned judge ({e}) — recreating alignment...")
    from mlflow.genai.judges import make_judge
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    JUDGE_MODEL = f"databricks:/{judge_model}"
    EMBEDDING_MODEL = "databricks:/databricks-gte-large-en"

    traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=50, return_type="list")
    rq_traces = [t for t in traces if any(a.name == "response_quality" for a in (getattr(t.info, "assessments", []) or []))]
    print(f"  Found {len(rq_traces)} traces with response_quality feedback")

    base_judge = make_judge(
        name="response_quality",
        instructions="Evaluate quality.\nQuestion: {{ inputs }}\nResponse: {{ outputs }}",
        feedback_value_type=bool,
        model=JUDGE_MODEL,
    )
    optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL, retrieval_k=3, embedding_model=EMBEDDING_MODEL)
    aligned_judge = base_judge.align(traces=rq_traces[:30], optimizer=optimizer)
    print(f"  Recreated aligned judge: {aligned_judge.name}")

# Trigger episodic memory initialization
try:
    _dummy = aligned_judge(
        inputs={"input": [{"role": "user", "content": "What is Unity Catalog?"}]},
        outputs={"response": "Unity Catalog is Databricks' data governance solution."},
    )
    print(f"  Dummy score: {_dummy}")
except Exception as e:
    print(f"  Dummy call: {type(e).__name__} (episodic memory should be initialized)")

if hasattr(aligned_judge, "_semantic_memory") and aligned_judge._semantic_memory:
    print(f"  Semantic memory: {len(aligned_judge._semantic_memory)} guidelines")
if hasattr(aligned_judge, "_episodic_memory") and aligned_judge._episodic_memory:
    print(f"  Episodic memory: {len(aligned_judge._episodic_memory)} examples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load current system prompt

# COMMAND ----------

# Load current system prompt from config.yaml
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
    _cfg = yaml.safe_load(f)
current_prompt = _cfg.get("system_prompt", "")
print(f"Loaded prompt from config.yaml ({len(current_prompt)} chars)")

# Try to register in Prompt Registry (optional — may not be available on all workspaces)
_prompt_registered = False
try:
    _prompt_obj = mlflow.genai.register_prompt(
        name=PROMPT_NAME, template=current_prompt,
        commit_message="Initial prompt from config.yaml",
    )
    _prompt_registered = True
    print(f"Registered prompt: {PROMPT_NAME}")
except Exception as e:
    print(f"Prompt Registry not available: {e}")
    print("Will run DSPy MIPROv2 without Prompt Registry (using direct prompt text)")

print(f"Prompt length: {len(current_prompt)} chars")
print(current_prompt[:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load evaluation dataset

# COMMAND ----------

import pandas as pd

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_df = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_df)} rows from {golden_table}")

# Convert to format for DSPy MIPROv2: list of dicts with 'inputs' and 'expectations'
eval_data = []
for _, row in eval_df.iterrows():
    query = row.get("request", row.get("input", ""))
    expected = row.get("expected_response", row.get("output", ""))
    entry = {"inputs": {"input": [{"role": "user", "content": query}]}}
    if expected:
        entry["expectations"] = {"expected_response": expected}
    eval_data.append(entry)

print(f"Converted to DSPy MIPROv2 format: {len(eval_data)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Define predict function

# COMMAND ----------

import requests as _requests

def predict_fn(**kwargs):
    """Query the deployed agent endpoint with the optimized prompt."""
    inputs = kwargs.get("input", kwargs.get("inputs", []))
    if isinstance(inputs, dict):
        inputs = inputs.get("input", [])

    system_prompt = current_prompt

    messages = [{"role": "system", "content": system_prompt}]
    if isinstance(inputs, list):
        messages.extend(inputs)
    else:
        messages.append({"role": "user", "content": str(inputs)})

    # Call via SDK api_client (handles auth internally, supports plain dict messages)
    from databricks.sdk import WorkspaceClient
    _w = WorkspaceClient()
    result = _w.api_client.do(
        "POST",
        f"/serving-endpoints/{judge_model}/invocations",
        body={"messages": messages, "max_tokens": 1024, "temperature": 0.1},
    )
    return result["choices"][0]["message"]["content"]

# Quick validation
_test = predict_fn(input=[{"role": "user", "content": "What is Delta Lake?"}])
print(f"Predict function validated ({len(_test)} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run DSPy MIPROv2 optimization rounds
# MAGIC
# MAGIC Each run produces a candidate prompt. Results are checkpointed to a Delta table
# MAGIC so optimization can be resumed if interrupted.

# COMMAND ----------

from mlflow.genai.optimize import DspyPromptOptimizer
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import time

# Checkpoint schema
_checkpoint_schema = StructType([
    StructField("agent_type", StringType()),
    StructField("run_idx", IntegerType()),
    StructField("initial_score", DoubleType()),
    StructField("final_score", DoubleType()),
    StructField("prompt_template", StringType()),
    StructField("elapsed_seconds", DoubleType()),
])

# Load existing checkpoints (resume support)
results = []
if spark.catalog.tableExists(CHECKPOINT_TABLE):
    _checkpoint_df = spark.table(CHECKPOINT_TABLE)
    results = [row.asDict() for row in _checkpoint_df.collect()]
    print(f"Loaded {len(results)} existing checkpoint rows")

completed_runs = {r["run_idx"] for r in results if r["agent_type"] == "base"}
print(f"Completed runs: {completed_runs}")
print(f"Remaining runs: {set(range(N_RUNS)) - completed_runs}")

# COMMAND ----------

# Run DSPy MIPROv2 optimization
# Ensure prompt is registered (needed for optimize_prompts)
if not _prompt_registered:
    try:
        _prompt_obj = mlflow.genai.register_prompt(name=PROMPT_NAME, template=current_prompt)
        _prompt_registered = True
        print(f"Registered prompt: {PROMPT_NAME} (uri={_prompt_obj.uri})")
    except Exception as e:
        print(f"Cannot register prompt for DSPy MIPROv2: {e}")
        print("DSPy MIPROv2 optimization requires Prompt Registry. Skipping.")
        import json
        dbutils.notebook.exit(json.dumps({"status": "skipped", "reason": "Prompt Registry unavailable"}))

# Use version number (not 'latest') to load prompt
prompt_uri = f"prompts:/{PROMPT_NAME}/1"
print(f"Prompt URI for DSPy MIPROv2: {prompt_uri}")

for run_idx in range(N_RUNS):
    if run_idx in completed_runs:
        print(f"\n[Run {run_idx}] Already completed — skipping")
        continue

    print(f"\n{'=' * 60}")
    print(f"  DSPy MIPROv2 Run {run_idx + 1}/{N_RUNS}")
    print(f"{'=' * 60}")

    t0 = time.time()
    try:
        result = mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=eval_data,
            prompt_uris=[prompt_uri],
            optimizer=DspyPromptOptimizer(
                max_metric_calls=MAX_METRIC_CALLS,
            ),
            scorers=[aligned_judge],
        )

        elapsed = time.time() - t0
        optimized_prompt = result.optimized_prompts[0]

        row = {
            "agent_type": "base",
            "run_idx": run_idx,
            "initial_score": float(result.initial_eval_score),
            "final_score": float(result.final_eval_score),
            "prompt_template": optimized_prompt.template,
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(row)

        # Checkpoint to Delta
        spark.createDataFrame([row], schema=_checkpoint_schema).write.mode("append").saveAsTable(CHECKPOINT_TABLE)

        print(f"  Initial: {result.initial_eval_score:.4f}")
        print(f"  Final:   {result.final_eval_score:.4f}")
        print(f"  Lift:    {result.final_eval_score / max(result.initial_eval_score, 0.001):.2f}x")
        print(f"  Time:    {elapsed:.0f}s")

    except Exception as e:
        print(f"  Run {run_idx} failed: {e}")
        elapsed = time.time() - t0
        results.append({
            "agent_type": "base", "run_idx": run_idx,
            "initial_score": None, "final_score": None,
            "prompt_template": None, "elapsed_seconds": round(elapsed, 1),
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Select best prompt and register

# COMMAND ----------

import pandas as pd

phase_results = [r for r in results if r["agent_type"] == "base" and r["final_score"] is not None]

if not phase_results:
    print("No successful optimization runs. Cannot select best prompt.")
    dbutils.notebook.exit(json.dumps({"status": "no_results"}))

best_run = max(phase_results, key=lambda r: r["final_score"] or 0)
best_prompt_text = best_run["prompt_template"]

dspy_mipro_df = pd.DataFrame(phase_results)
dspy_mipro_df["lift"] = dspy_mipro_df["final_score"] / dspy_mipro_df["initial_score"].clip(lower=0.001)

print("DSPy MIPROv2 OPTIMIZATION RESULTS")
print("=" * 90)
print(dspy_mipro_df[["run_idx", "initial_score", "final_score", "lift", "elapsed_seconds"]].to_string(index=False, float_format="%.4f"))

print(f"\n  Initial mean (1-5): {dspy_mipro_df['initial_score'].mean() * 5:.2f}")
print(f"  Final mean (1-5):   {dspy_mipro_df['final_score'].mean() * 5:.2f}")
print(f"  Best run: {best_run['run_idx']} (score: {best_run['initial_score']:.3f} -> {best_run['final_score']:.3f})")

# Register best prompt
new_prompt = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=best_prompt_text,
    commit_message=(
        f"Best prompt from DSPy MIPROv2 (run={best_run['run_idx']}, "
        f"score: {best_run['initial_score']:.3f} -> {best_run['final_score']:.3f}, "
        f"judge: {aligned_judge_name})"
    ),
    tags={"experiment": "dspy_mipro_optimization", "run_idx": str(best_run["run_idx"])},
)
print(f"\nRegistered optimized prompt: {PROMPT_NAME} (version {new_prompt.version})")
print(f"First 300 chars:\n{best_prompt_text[:300]}...")

# Update production alias to point to the optimized prompt
mlflow.genai.set_prompt_alias(
    name=PROMPT_NAME,
    alias="production",
    version=new_prompt.version,
)
print(f"Updated @production alias → v{new_prompt.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="prompt_optimization", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("dspy_mipro_optimization", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(eval_data), output_summary={
    "n_runs": N_RUNS,
    "max_metric_calls": MAX_METRIC_CALLS,
    "best_run": best_run["run_idx"],
    "initial_score": best_run["initial_score"],
    "final_score": best_run["final_score"],
    "prompt_name": PROMPT_NAME,
    "checkpoint_table": CHECKPOINT_TABLE,
})

pipeline.end(status="COMPLETED")

dbutils.notebook.exit(json.dumps({
    "status": "completed",
    "best_run": best_run["run_idx"],
    "initial_score": best_run["initial_score"],
    "final_score": best_run["final_score"],
    "prompt_name": PROMPT_NAME,
}))

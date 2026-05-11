# Databricks notebook source
# MAGIC %md
# MAGIC # Iterative Improvement — MemAlign + GEPA
# MAGIC Collect expert feedback, align the LLM judge, and optimize the system prompt.
# MAGIC
# MAGIC **Reference:** [Self-Optimizing Chatbot](https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks)
# MAGIC
# MAGIC **APIs used (MLflow 3.4+):**
# MAGIC - `mlflow.genai.judges.make_judge()` — create custom LLM judge
# MAGIC - `judge.align(traces, GEPAAlignmentOptimizer)` — align judge with expert feedback
# MAGIC - `mlflow.genai.optimize_prompts()` — GEPA prompt optimization (MLflow >= 3.5)
# MAGIC
# MAGIC **Environment:** Requires `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE` env vars
# MAGIC for litellm routing. Auto-configured by `setup_databricks_env()` on cluster.
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Load agent config and create LLM judge with `make_judge()`
# MAGIC 2. Collect expert-labeled traces from MLflow
# MAGIC 3. Align judge with expert preferences using GEPA (LLM reflection, no embeddings)
# MAGIC 4. Optimize system prompt using GEPA prompt optimizer
# MAGIC 5. Compare optimized prompt vs baseline

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("judge_name", "response_quality")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_name = dbutils.widgets.get("judge_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

# Ensure MLflow 3.x with genai module (needed for make_judge, GEPA, optimize_prompts)
import subprocess, os

# Install from Mastercard volume (air-gapped) or PyPI
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
_wheels_path = _vol_path if os.path.exists(_vol_path) else None

if _wheels_path:
    print(f"Installing from: {_wheels_path}")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "dspy", "gepa", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.5", "dspy>=2.6", "gepa", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_name = dbutils.widgets.get("judge_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load current agent config

# COMMAND ----------

import yaml

nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
    agent_config = yaml.safe_load(f)

current_prompt = agent_config.get("system_prompt", "")
print(f"Current system prompt ({len(current_prompt)} chars):")
print(current_prompt[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create LLM judge and collect expert feedback

# COMMAND ----------

from framework.optimization.prompt_optimizer import IterativeOptimizer

optimizer = IterativeOptimizer(
    agent_name=agent_name,
    experiment_name=f"/Users/{spark.sql('SELECT current_user()').first()[0]}/{agent_name}",
)

# Create a custom judge using make_judge()
judge = optimizer.create_judge(judge_name=judge_name)
print(f"Created judge: '{judge_name}'")

# Collect traces with human feedback matching the judge name
# Search both the experiment AND the serving endpoint model traces (Review App feedback)
model_name = f"{catalog}.{schema}.{agent_name}"
feedback = optimizer.collect_expert_feedback(
    max_traces=200, judge_name=judge_name, model_name=model_name,
)
print(f"\nExpert-labeled traces: {len(feedback)}")

if feedback:
    for t in feedback[:3]:
        print(f"\n  Trace: {t.info.request_id}")
        assessments = [a for a in t.info.assessments if a.name == judge_name]
        print(f"  Assessments matching '{judge_name}': {len(assessments)}")
else:
    print(f"\nNo traces found with '{judge_name}' feedback.")
    print("Label traces in the MLflow UI first:")
    print(f"  1. Open MLflow experiment for {agent_name}")
    print(f"  2. Click on a trace → Add assessment")
    print(f"  3. Use name '{judge_name}' (must match judge name for alignment)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Align judge with expert preferences (GEPA)
# MAGIC
# MAGIC `judge.align(traces, GEPAAlignmentOptimizer)` uses LLM-driven reflection
# MAGIC to calibrate the judge to score like human domain experts.
# MAGIC No embedding model needed (unlike MemAlign). Requires at least 10 labeled traces.

# COMMAND ----------

alignment_result = optimizer.align_judge(traces=feedback, optimizer_type="gepa")
print(f"Judge alignment: {alignment_result['status']}")

if alignment_result.get("num_traces"):
    print(f"  Traces used: {alignment_result['num_traces']}")

if alignment_result["status"] == "aligned" and optimizer._aligned_judge:
    print(f"\nAligned judge instructions (first 500 chars):")
    print(optimizer._aligned_judge.instructions[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load evaluation dataset

# COMMAND ----------

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_df = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_df)} rows from {golden_table}")

# Convert to format required by optimize_prompts: list of dicts with 'inputs' and 'expectations'
eval_dataset = [
    {
        "inputs": {"question": row.get("request", row.get("input", ""))},
        "expectations": {"expected_response": row.get("expected_response", row.get("output", ""))},
    }
    for _, row in eval_df.iterrows()
]
print(f"Converted to optimize_prompts format: {len(eval_dataset)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Optimize system prompt (GEPA)
# MAGIC
# MAGIC Uses `mlflow.genai.optimize_prompts()` with `GepaPromptOptimizer` to
# MAGIC generate and evaluate candidate prompt variations. The aligned judge
# MAGIC from step 3 is used as the scorer.

# COMMAND ----------

optimization_result = optimizer.optimize_prompt(
    current_prompt=current_prompt,
    eval_dataset=eval_dataset,
    prompt_name=f"{agent_name}_system_prompt",
)

print(f"Optimization: {optimization_result['status']}")
print(f"Baseline score: {optimization_result.get('baseline_score')}")
if optimization_result.get("optimized_score"):
    print(f"Optimized score: {optimization_result['optimized_score']}")
    print(f"Improvement: {optimization_result['improvement']:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Review optimized prompt

# COMMAND ----------

if optimization_result.get("optimized_prompt"):
    print("=== Optimized System Prompt ===")
    print(optimization_result["optimized_prompt"])
    print(f"\nPrompt URI: {optimization_result.get('prompt_uri')}")
    print("\n=== Next Steps ===")
    print("1. Review the optimized prompt above")
    print("2. If satisfied, update config.yaml with the new prompt")
    print("3. Re-register the agent (Agent.py notebook)")
    print("4. Run evaluation to verify improvement")
elif optimization_result.get("status") == "skipped":
    print(f"Prompt optimization skipped: {optimization_result.get('reason')}")
    print("\nRequires MLflow >= 3.5 with mlflow.genai.optimize_prompts()")
else:
    print("No optimized prompt generated.")
    print("Ensure you have:")
    print("  1. At least 10 expert-labeled traces")
    print("  2. MLflow >= 3.5 installed")
    print("  3. An aligned judge (step 3 must succeed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log to audit

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="iterative_improvement", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("create_judge", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED",
                  output_summary={"judge_name": judge_name, "model": optimizer.judge_model})

step = pipeline.start_step("collect_feedback", step_order=2, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(feedback),
                  output_summary={"labeled_traces": len(feedback)})

step = pipeline.start_step("align_judge", step_order=3, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", output_summary=alignment_result)

step = pipeline.start_step("optimize_prompt", step_order=4, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "baseline_score": optimization_result.get("baseline_score"),
    "optimized_score": optimization_result.get("optimized_score"),
    "improvement": optimization_result.get("improvement"),
})

pipeline.end(status="COMPLETED")
print("Optimization cycle logged to audit tables")

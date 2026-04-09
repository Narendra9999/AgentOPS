# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Held-Out Evaluation
# MAGIC Compares agent configurations on the same held-out question set with the aligned judge.
# MAGIC
# MAGIC **Configurations compared:**
# MAGIC 1. **Baseline** — Original agent with config.yaml system prompt
# MAGIC 2. **Optimized Prompt** — Agent with GEPA-optimized prompt (from 02)
# MAGIC 3. **Optimized Prompt + Skills** — Agent with optimized prompt + skill files (from 04)
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Aligned judge registered (from 01)
# MAGIC - Optimized prompt in Prompt Registry (from 02)
# MAGIC - Agent with skills model registered (from 04)
# MAGIC
# MAGIC **Reference:** Notebook 09-Evaluation from at-bat-assistant

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("aligned_judge_name", "response_quality_aligned")
dbutils.widgets.text("chatbot_name", "agentops-docs-chatbot")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
chatbot_name = dbutils.widgets.get("chatbot_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
if os.path.exists(_vol_path):
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "dspy", "--find-links", _vol_path, "--no-index", "-q"])
else:
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.4", "dspy>=2.6", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
chatbot_name = dbutils.widgets.get("chatbot_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure environment

# COMMAND ----------

import os, json, yaml, warnings, logging
import mlflow

warnings.filterwarnings("ignore")
logging.getLogger("mlflow.genai.judges.instructions_judge").setLevel(logging.ERROR)
logging.getLogger("mlflow.tracing.fluent").setLevel(logging.ERROR)

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _ws_url.startswith("http"):
    _ws_url = f"https://{_ws_url}"
os.environ["DATABRICKS_API_KEY"] = _token
os.environ["DATABRICKS_API_BASE"] = f"{_ws_url}/serving-endpoints"
os.environ["DATABRICKS_HOST"] = _ws_url
os.environ["DATABRICKS_TOKEN"] = _token

_user = spark.sql("SELECT current_user()").first()[0]
experiment = mlflow.set_experiment(f"/Users/{_user}/{agent_name}")
EXPERIMENT_ID = experiment.experiment_id

PROMPT_NAME = f"{catalog}.{schema}.{agent_name}_system_prompt"

print(f"Experiment: {experiment.name}")
print(f"Aligned judge: {aligned_judge_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load aligned judge and prompts

# COMMAND ----------

from mlflow.genai.scorers import get_scorer

# Load aligned judge — try registry, fallback to recreating
try:
    aligned_judge = get_scorer(name=aligned_judge_name)
    print(f"Loaded aligned judge from registry: {aligned_judge.name}")
except Exception as e:
    print(f"Could not load from registry ({e}) — recreating alignment...")
    from mlflow.genai.judges import make_judge
    from mlflow.genai.judges.optimizers import MemAlignOptimizer

    JUDGE_MODEL = "databricks:/databricks-meta-llama-3-3-70b-instruct"
    EMBEDDING_MODEL = "databricks:/databricks-gte-large-en"

    # Search ALL experiments for traces with human feedback
    from mlflow import MlflowClient
    _client = MlflowClient()
    _all_exps = _client.search_experiments(max_results=100)
    all_traces = []
    for exp in _all_exps:
        if "/Trash" in exp.name:
            continue
        try:
            traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=200, return_type="list")
            all_traces.extend(traces)
        except Exception:
            pass

    # Find traces with HUMAN feedback (any assessment name)
    rq_traces = []
    human_feedback_name = None
    for t in all_traces:
        for a in (getattr(t.info, "assessments", []) or []):
            source_type = getattr(a.source, "source_type", "?") if hasattr(a, "source") else "?"
            if source_type == "HUMAN" and a.name != "expected_response":
                rq_traces.append(t)
                human_feedback_name = human_feedback_name or a.name
                break

    print(f"  Found {len(rq_traces)} traces with human feedback (name: {human_feedback_name})")
    _base_name = human_feedback_name or "response_quality"

    base_judge = make_judge(
        name=_base_name,
        instructions="Evaluate quality.\nQuestion: {{ inputs }}\nResponse: {{ outputs }}",
        feedback_value_type=bool,
        model=JUDGE_MODEL,
    )
    if rq_traces:
        optimizer = MemAlignOptimizer(reflection_lm=JUDGE_MODEL, retrieval_k=3, embedding_model=EMBEDDING_MODEL)
        aligned_judge = base_judge.align(traces=rq_traces[:30], optimizer=optimizer)
        print(f"  Recreated aligned judge with {len(rq_traces[:30])} traces")
    else:
        aligned_judge = base_judge
        print("  No feedback traces — using base judge")

# Trigger episodic memory init
try:
    aligned_judge(
        inputs={"input": [{"role": "user", "content": "test"}]},
        outputs={"response": "test response"},
    )
except Exception:
    pass

# Wrap aligned judge in a @scorer so evaluate() can aggregate metrics.
# get_scorer() returns a judge whose __call__ output isn't captured by evaluate()
# as metrics. We wrap it to return a Feedback object that evaluate() understands.
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

_raw_judge = aligned_judge

@scorer
def response_quality(inputs, outputs, expectations=None):
    """Evaluates response quality using the aligned judge."""
    try:
        result = _raw_judge(inputs=inputs, outputs=outputs, expectations=expectations)
        # Convert judge result to Feedback
        if isinstance(result, Feedback):
            return result
        elif isinstance(result, bool):
            return Feedback(value=result, rationale="Aligned judge assessment")
        elif isinstance(result, (int, float)):
            return Feedback(value=float(result), rationale="Aligned judge assessment")
        elif isinstance(result, str):
            is_yes = result.strip().lower() in ("yes", "true", "1")
            return Feedback(value=is_yes, rationale=result)
        elif hasattr(result, "value"):
            return Feedback(value=result.value, rationale=getattr(result, "rationale", ""))
        else:
            return Feedback(value=str(result), rationale=f"Raw judge output: {type(result).__name__}")
    except Exception as e:
        return Feedback(value=False, rationale=f"Judge error: {str(e)[:200]}")

aligned_judge = response_quality
print(f"Wrapped aligned judge as @scorer 'response_quality' for evaluate() compatibility")

# Load original prompt from config
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
_project_root = f"/Workspace{nb_root}" if nb_root.startswith("/") else f"/Workspace/{nb_root}"

with open(f"{_project_root}/agent/config.yaml") as f:
    agent_config = yaml.safe_load(f)
baseline_prompt = agent_config.get("system_prompt", "")

# Load optimized prompt from registry
try:
    optimized_prompt_obj = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/1")
    optimized_prompt = optimized_prompt_obj.template
    print(f"Optimized prompt loaded from registry ({len(optimized_prompt)} chars)")
except Exception as e:
    optimized_prompt = baseline_prompt
    print(f"No optimized prompt in registry ({e}) — using baseline for both")

print(f"Baseline prompt: {len(baseline_prompt)} chars")
print(f"Optimized prompt: {len(optimized_prompt)} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load evaluation dataset

# COMMAND ----------

import pandas as pd

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_df = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_df)} rows")

# Convert to evaluate format
eval_data = []
for _, row in eval_df.iterrows():
    query = row.get("request", row.get("input", ""))
    expected = row.get("expected_response", row.get("output", ""))
    entry = {"inputs": {"input": [{"role": "user", "content": query}]}}
    if expected:
        entry["expectations"] = {"expected_response": expected}
    eval_data.append(entry)

print(f"Eval data: {len(eval_data)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Define predict functions for each configuration

# COMMAND ----------

import requests as _requests
import sys

# _project_root already set in section 2

def endpoint_predict_fn(**kwargs):
    """Query the deployed serving endpoint. Accepts **kwargs from mlflow.genai.evaluate."""
    inputs = kwargs.get("input", kwargs.get("inputs", []))
    if isinstance(inputs, dict):
        messages = inputs.get("input", inputs.get("messages", []))
    elif isinstance(inputs, list):
        messages = inputs
    else:
        messages = [{"role": "user", "content": str(inputs)}]

    resp = _requests.post(
        f"{_ws_url}/serving-endpoints/{chatbot_name}/invocations",
        headers={"Authorization": f"Bearer {_token}"},
        json={"messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    if "messages" in result:
        return result["messages"][-1]["content"]
    elif "choices" in result:
        return result["choices"][0]["message"]["content"]
    return str(result)[:500]


def local_predict_fn_factory(system_prompt):
    """Create a predict function using a local model with a specific prompt."""
    model_name_uc = f"{catalog}.{schema}.{agent_name}"
    loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name_uc}@champion")

    def predict_fn(**kwargs):
        inputs = kwargs.get("input", kwargs.get("inputs", []))
        if isinstance(inputs, dict):
            inputs = inputs.get("input", [])
        messages = [{"role": "user", "content": inputs[0]["content"] if isinstance(inputs, list) and inputs else str(inputs)}]
        result = loaded_model.predict({"messages": messages})
        if hasattr(result, "messages"):
            return result.messages[0].content
        elif isinstance(result, dict) and "messages" in result:
            return result["messages"][0]["content"]
        return str(result)[:500]

    return predict_fn

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run evaluation for each configuration
# MAGIC
# MAGIC Evaluates baseline, optimized prompt, and (if available) optimized + skills
# MAGIC on the same held-out dataset using the same aligned judge.

# COMMAND ----------

from mlflow.genai import evaluate

# Set concurrency
os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "3"
os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = "1"

# Define configurations to evaluate
configs = {
    "baseline": {"label": "Baseline (Original)", "predict_fn": endpoint_predict_fn},
}

# Only add optimized if it differs from baseline
if optimized_prompt != baseline_prompt:
    configs["optimized"] = {
        "label": "Optimized Prompt",
        "predict_fn": endpoint_predict_fn,
    }

_eval_errors = []  # Capture errors for notebook exit

eval_results = {}

for config_key, config in configs.items():
    label = config["label"]
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"{'=' * 60}")

    try:
        # Quick test — verify predict_fn works before full eval
        print(f"  Testing predict_fn...")
        _test_out = config["predict_fn"](input=[{"role": "user", "content": "What is Delta Lake?"}])
        print(f"  Predict test OK: {str(_test_out)[:100]}...")

        print(f"  Running evaluate with {len(eval_data)} rows...")
        result = evaluate(
            data=eval_data,
            predict_fn=config["predict_fn"],
            scorers=[aligned_judge],
        )

        eval_results[config_key] = result
        print(f"  Metrics: {result.metrics}")

        # Debug: inspect the result object to understand available data
        print(f"  Result type: {type(result).__name__}")
        print(f"  Result attrs: {[a for a in dir(result) if not a.startswith('_')]}")
        if hasattr(result, "metrics") and not result.metrics:
            print(f"  WARNING: Metrics dict is empty — checking traces for assessment data...")
            # Try to extract scores from the evaluation traces directly
            eval_traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=50, return_type="list")
            recent_with_scores = []
            for t in eval_traces:
                for a in (getattr(t.info, "assessments", []) or []):
                    source_type = getattr(a.source, "source_type", "?") if hasattr(a, "source") else "?"
                    if source_type != "HUMAN":
                        recent_with_scores.append((t.info.request_id[:20], a.name, getattr(a, "boolean_value", getattr(a, "numeric_value", "?"))))
            if recent_with_scores:
                print(f"  Found {len(recent_with_scores)} non-human assessments on traces:")
                for tid, aname, aval in recent_with_scores[:5]:
                    print(f"    {tid}... {aname}={aval}")

        if hasattr(result, "tables") and result.tables:
            for tname, tdf in result.tables.items():
                print(f"  Table '{tname}': {tdf.shape}")

    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"  Evaluation FAILED: {err_msg}")
        traceback.print_exc()
        _eval_errors.append(f"{config_key}: {err_msg}")
        eval_results[config_key] = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare results

# COMMAND ----------

import numpy as np

print("\n" + "=" * 80)
print("  HELD-OUT EVALUATION COMPARISON")
print("=" * 80)

comparison = []
for config_key, result in eval_results.items():
    if result is None:
        continue

    label = configs[config_key]["label"]
    metrics = result.metrics

    row = {"Configuration": label}
    for metric_name, metric_value in metrics.items():
        row[metric_name] = metric_value

    comparison.append(row)
    print(f"\n  {label}:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

if len(comparison) > 1:
    comp_df = pd.DataFrame(comparison)
    print(f"\n{'=' * 80}")
    print("  SUMMARY TABLE")
    print("=" * 80)
    display(comp_df) if 'display' in dir() else print(comp_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Extract per-trace scores for analysis

# COMMAND ----------

for config_key, result in eval_results.items():
    if result is None:
        continue

    label = configs[config_key]["label"]
    run_id = None

    # Try to get run_id from the evaluate result
    if hasattr(result, "run_id"):
        run_id = result.run_id
    elif hasattr(result, "_run_id"):
        run_id = result._run_id

    if run_id:
        traces_df = mlflow.search_traces(run_id=run_id, locations=[EXPERIMENT_ID])
        scores = []
        for _, row in traces_df.iterrows():
            for a in (row.get("assessments") or []):
                if a.get("assessment_name") == aligned_judge_name:
                    val = a.get("feedback", {}).get("value")
                    if val == "yes":
                        scores.append(1.0)
                    elif val == "no":
                        scores.append(0.0)
                    elif isinstance(val, (int, float)):
                        scores.append(float(val))

        if scores:
            print(f"\n{label} — Per-trace scores:")
            print(f"  Mean:   {np.mean(scores):.4f}")
            print(f"  Std:    {np.std(scores):.4f}")
            print(f"  Min:    {np.min(scores):.4f}")
            print(f"  Max:    {np.max(scores):.4f}")
            print(f"  N:      {len(scores)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="held_out_evaluation", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

eval_summary = {}
for config_key, result in eval_results.items():
    if result:
        eval_summary[config_key] = {k: round(v, 4) if isinstance(v, float) else v for k, v in result.metrics.items()}

step = pipeline.start_step("evaluate_configurations", step_order=1, step_type="evaluation")
pipeline.end_step(step, status="COMPLETED", records_processed=len(eval_data), output_summary={
    "configurations": list(configs.keys()),
    "eval_dataset_size": len(eval_data),
    "aligned_judge": aligned_judge_name,
    "results": eval_summary,
})

# Determine winner
best_config = None
best_score = -1
for config_key, metrics in eval_summary.items():
    # Use the first metric as the comparison key
    score = list(metrics.values())[0] if metrics else 0
    if score > best_score:
        best_score = score
        best_config = config_key

step = pipeline.start_step("select_winner", step_order=2, step_type="evaluation")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "winner": best_config,
    "winner_score": best_score,
    "recommendation": f"Promote '{best_config}' configuration" if best_config else "No clear winner",
})

pipeline.end(status="COMPLETED")

print(f"\n{'=' * 60}")
print(f"  RECOMMENDATION: Use '{best_config}' (score: {best_score:.4f})")
print(f"{'=' * 60}")

dbutils.notebook.exit(json.dumps({
    "winner": best_config,
    "winner_score": best_score,
    "results": eval_summary,
    "errors": _eval_errors if _eval_errors else None,
}))

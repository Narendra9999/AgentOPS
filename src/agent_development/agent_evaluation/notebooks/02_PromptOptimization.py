# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Prompt Optimization with DSPy MIPROv2
# MAGIC Runs DSPy MIPROv2 iterative prompt optimization using the aligned judge as the metric.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Aligned judge registered (from 05_JudgeAlignment)
# MAGIC - Evaluation dataset in UC table
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads the aligned judge from the experiment
# MAGIC 2. Loads the current system prompt from config / MLflow Prompt Registry
# MAGIC 3. Configures DSPy with Databricks LLM endpoint
# MAGIC 4. Runs MIPROv2 iterative optimization (generates candidates, evaluates, selects best)
# MAGIC 5. Registers the best prompt in MLflow Prompt Registry
# MAGIC
# MAGIC **Optimizer:** DSPy MIPROv2 (iterative, multi-round — no GEPA dependency)

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-120b")
dbutils.widgets.text("judge_model", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("aligned_judge_name", "response_quality_aligned")
dbutils.widgets.text("num_candidates", "7")
dbutils.widgets.text("max_bootstrapped_demos", "3")
dbutils.widgets.text("max_labeled_demos", "5")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
NUM_CANDIDATES = int(dbutils.widgets.get("num_candidates"))
MAX_BOOTSTRAPPED_DEMOS = int(dbutils.widgets.get("max_bootstrapped_demos"))
MAX_LABELED_DEMOS = int(dbutils.widgets.get("max_labeled_demos"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os

# Install from Mastercard volume (air-gapped) or PyPI
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
_wheels_path = _vol_path if os.path.exists(_vol_path) else None

if _wheels_path:
    print(f"Installing from: {_wheels_path}")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "dspy", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.5", "dspy>=2.6", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
NUM_CANDIDATES = int(dbutils.widgets.get("num_candidates"))
MAX_BOOTSTRAPPED_DEMOS = int(dbutils.widgets.get("max_bootstrapped_demos"))
MAX_LABELED_DEMOS = int(dbutils.widgets.get("max_labeled_demos"))

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

PROMPT_NAME = f"{catalog}.{schema}.{agent_name}_system_prompt"
CHECKPOINT_TABLE = f"{catalog}.{schema}.dspy_mipro_checkpoint"

# Set experiment
_user = spark.sql("SELECT current_user()").first()[0]
experiment = mlflow.set_experiment(f"/Users/{_user}/{agent_name}")
EXPERIMENT_ID = experiment.experiment_id

print(f"Experiment: {experiment.name} (ID: {EXPERIMENT_ID})")
print(f"LLM endpoint: {llm_endpoint}")
print(f"Judge model: {judge_model}")
print(f"Prompt name: {PROMPT_NAME}")
print(f"MIPROv2 config: {NUM_CANDIDATES} candidates, {MAX_BOOTSTRAPPED_DEMOS} bootstrapped demos, {MAX_LABELED_DEMOS} labeled demos")

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
print(current_prompt[:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load evaluation dataset and convert to DSPy format

# COMMAND ----------

import pandas as pd
import dspy

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_df = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_df)} rows from {golden_table}")

# Convert to DSPy Examples
dspy_trainset = []
for _, row in eval_df.iterrows():
    question = row.get("request", row.get("input", ""))
    expected = row.get("expected_response", row.get("output", ""))
    ex = dspy.Example(question=question, expected_answer=expected).with_inputs("question")
    dspy_trainset.append(ex)

print(f"Converted to DSPy trainset: {len(dspy_trainset)} examples")
print(f"Sample: {dspy_trainset[0].question[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configure DSPy with Databricks LLM

# COMMAND ----------

# Configure DSPy to route through Databricks FMAPI
lm = dspy.LM(
    f"databricks/{llm_endpoint}",
    api_base=f"{_ws_url}/serving-endpoints",
    api_key=_token,
    max_tokens=1024,
    temperature=0.1,
)
dspy.configure(lm=lm)
print(f"DSPy configured with: databricks/{llm_endpoint}")

# Quick validation
_test_pred = lm("What is Delta Lake? Answer in one sentence.")
print(f"LM validation: {str(_test_pred)[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Define DSPy module and metric

# COMMAND ----------

class DocsQA(dspy.Module):
    """DSPy module that wraps our documentation QA agent for optimization."""

    def __init__(self, system_prompt):
        super().__init__()
        # The signature defines the task — MIPROv2 will optimize the instructions
        self.generate = dspy.ChainOfThought(
            dspy.Signature(
                "question -> answer",
                instructions=system_prompt,
            )
        )

    def forward(self, question):
        return self.generate(question=question)


def eval_metric(example, prediction, trace=None):
    """Evaluate a prediction using the aligned judge.

    Returns True/False for MIPROv2's optimization loop.
    """
    try:
        score = aligned_judge(
            inputs={"input": [{"role": "user", "content": example.question}]},
            outputs={"response": prediction.answer},
        )
        # aligned_judge returns bool or numeric — normalize to bool
        if isinstance(score, bool):
            return score
        if isinstance(score, (int, float)):
            return score > 0.5
        # Handle dict-like score objects
        val = getattr(score, "value", score)
        if isinstance(val, bool):
            return val
        return bool(val)
    except Exception as e:
        print(f"  Metric error: {e}")
        return False


# Validate metric with one example
_agent = DocsQA(system_prompt=current_prompt)
_pred = _agent(question=dspy_trainset[0].question)
_score = eval_metric(dspy_trainset[0], _pred)
print(f"Metric validation: question='{dspy_trainset[0].question[:50]}...', score={_score}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Run MIPROv2 optimization
# MAGIC
# MAGIC MIPROv2 iteratively:
# MAGIC 1. Generates candidate instruction variations
# MAGIC 2. Bootstraps few-shot demonstrations from your eval data
# MAGIC 3. Evaluates each candidate against the metric
# MAGIC 4. Selects the best-performing combination of instructions + demos

# COMMAND ----------

from dspy.teleprompt import MIPROv2
import time

agent = DocsQA(system_prompt=current_prompt)

# Evaluate baseline before optimization
print("=== Baseline Evaluation ===")
baseline_scores = []
for ex in dspy_trainset:
    try:
        pred = agent(question=ex.question)
        score = eval_metric(ex, pred)
        baseline_scores.append(1.0 if score else 0.0)
    except Exception:
        baseline_scores.append(0.0)
baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
print(f"Baseline score: {baseline_score:.3f} ({sum(baseline_scores):.0f}/{len(baseline_scores)} passed)")

# COMMAND ----------

print(f"\n{'=' * 60}")
print(f"  MIPROv2 Optimization")
print(f"  Candidates: {NUM_CANDIDATES} | Bootstrapped demos: {MAX_BOOTSTRAPPED_DEMOS} | Labeled demos: {MAX_LABELED_DEMOS}")
print(f"{'=' * 60}\n")

t0 = time.time()

optimizer = MIPROv2(
    metric=eval_metric,
    num_candidates=NUM_CANDIDATES,
    num_threads=2,
    verbose=True,
)

with mlflow.start_run(run_name="dspy_mipro_optimization"):
    mlflow.log_params({
        "optimizer": "MIPROv2",
        "num_candidates": NUM_CANDIDATES,
        "max_bootstrapped_demos": MAX_BOOTSTRAPPED_DEMOS,
        "max_labeled_demos": MAX_LABELED_DEMOS,
        "trainset_size": len(dspy_trainset),
        "llm_endpoint": llm_endpoint,
        "judge_model": judge_model,
        "baseline_score": baseline_score,
    })

    optimized_agent = optimizer.compile(
        agent,
        trainset=dspy_trainset,
        max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
        max_labeled_demos=MAX_LABELED_DEMOS,
    )

    elapsed = time.time() - t0
    print(f"\nOptimization completed in {elapsed:.0f}s")

    # Evaluate optimized agent
    print("\n=== Optimized Agent Evaluation ===")
    optimized_scores = []
    for ex in dspy_trainset:
        try:
            pred = optimized_agent(question=ex.question)
            score = eval_metric(ex, pred)
            optimized_scores.append(1.0 if score else 0.0)
        except Exception:
            optimized_scores.append(0.0)
    optimized_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0

    mlflow.log_metrics({
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - baseline_score,
        "optimization_time_s": elapsed,
    })

    print(f"Baseline:  {baseline_score:.3f}")
    print(f"Optimized: {optimized_score:.3f}")
    print(f"Lift:      {optimized_score - baseline_score:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Extract optimized prompt and register

# COMMAND ----------

# Extract the optimized instructions from the compiled DSPy module
optimized_instructions = optimized_agent.generate.signature.instructions
optimized_demos = []

# Extract few-shot demonstrations if any were selected
if hasattr(optimized_agent.generate, "demos") and optimized_agent.generate.demos:
    for demo in optimized_agent.generate.demos:
        optimized_demos.append({
            "question": getattr(demo, "question", ""),
            "answer": getattr(demo, "answer", ""),
        })

print(f"Optimized instructions ({len(optimized_instructions)} chars):")
print(optimized_instructions[:500] + "...")
print(f"\nFew-shot demos: {len(optimized_demos)}")

# Build final prompt — instructions + demos as examples
final_prompt = optimized_instructions
if optimized_demos:
    examples_text = "\n\nHere are some example interactions:\n"
    for i, demo in enumerate(optimized_demos, 1):
        examples_text += f"\nExample {i}:\nUser: {demo['question']}\nAssistant: {demo['answer'][:300]}\n"
    final_prompt += examples_text

print(f"\nFinal prompt length: {len(final_prompt)} chars")

# COMMAND ----------

# Checkpoint results to Delta
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

_checkpoint_schema = StructType([
    StructField("optimizer", StringType()),
    StructField("baseline_score", DoubleType()),
    StructField("optimized_score", DoubleType()),
    StructField("improvement", DoubleType()),
    StructField("prompt_template", StringType()),
    StructField("num_demos", DoubleType()),
    StructField("elapsed_seconds", DoubleType()),
])

checkpoint_row = {
    "optimizer": "MIPROv2",
    "baseline_score": baseline_score,
    "optimized_score": optimized_score,
    "improvement": optimized_score - baseline_score,
    "prompt_template": final_prompt,
    "num_demos": float(len(optimized_demos)),
    "elapsed_seconds": round(elapsed, 1),
}
spark.createDataFrame([checkpoint_row], schema=_checkpoint_schema).write.mode("append").saveAsTable(CHECKPOINT_TABLE)
print(f"Checkpointed to {CHECKPOINT_TABLE}")

# COMMAND ----------

# Register in MLflow Prompt Registry (if improvement > 0)
if optimized_score > baseline_score:
    try:
        new_prompt = mlflow.genai.register_prompt(
            name=PROMPT_NAME,
            template=final_prompt,
            commit_message=(
                f"DSPy MIPROv2 optimized prompt "
                f"(score: {baseline_score:.3f} -> {optimized_score:.3f}, "
                f"+{optimized_score - baseline_score:.3f}, "
                f"demos: {len(optimized_demos)}, judge: {aligned_judge_name})"
            ),
            tags={"experiment": "dspy_mipro", "optimizer": "MIPROv2"},
        )
        print(f"Registered optimized prompt: {PROMPT_NAME} (version {new_prompt.version})")

        # Update production alias
        mlflow.genai.set_prompt_alias(
            name=PROMPT_NAME,
            alias="production",
            version=new_prompt.version,
        )
        print(f"Updated @production alias -> v{new_prompt.version}")
    except Exception as e:
        print(f"Could not register prompt: {e}")
        print("Prompt text saved to checkpoint table — register manually if needed.")
else:
    print(f"No improvement (baseline={baseline_score:.3f}, optimized={optimized_score:.3f})")
    print("Keeping current prompt. Optimized prompt saved to checkpoint table for review.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="prompt_optimization", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("dspy_mipro_optimization", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(dspy_trainset), output_summary={
    "optimizer": "MIPROv2",
    "num_candidates": NUM_CANDIDATES,
    "max_bootstrapped_demos": MAX_BOOTSTRAPPED_DEMOS,
    "max_labeled_demos": MAX_LABELED_DEMOS,
    "baseline_score": baseline_score,
    "optimized_score": optimized_score,
    "improvement": optimized_score - baseline_score,
    "num_demos": len(optimized_demos),
    "prompt_name": PROMPT_NAME,
    "checkpoint_table": CHECKPOINT_TABLE,
})

pipeline.end(status="COMPLETED")

dbutils.notebook.exit(json.dumps({
    "status": "completed",
    "optimizer": "MIPROv2",
    "baseline_score": baseline_score,
    "optimized_score": optimized_score,
    "improvement": optimized_score - baseline_score,
    "prompt_name": PROMPT_NAME,
}))

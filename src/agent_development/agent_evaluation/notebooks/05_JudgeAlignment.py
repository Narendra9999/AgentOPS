# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Judge Alignment with MemAlign
# MAGIC Aligns the evaluation judge with SME (Subject Matter Expert) feedback using MemAlign.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Evaluation traces with expert labels (from Review App or `mlflow.log_feedback()`)
# MAGIC - At least 10 labeled traces for alignment
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads evaluation traces with expert feedback
# MAGIC 2. Creates a MemAlignOptimizer to distill guidelines from expert annotations
# MAGIC 3. Aligns the base judge → aligned judge reflecting organizational preferences
# MAGIC 4. Inspects distilled semantic memory (guidelines) and episodic memory (examples)
# MAGIC 5. Registers the aligned judge for use in prompt optimization
# MAGIC
# MAGIC **Reference:** [Self-Optimizing Chatbot](https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks)

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("judge_name", "response_quality")
dbutils.widgets.text("judge_model", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("embedding_model", "databricks-gte-large-en")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_name = dbutils.widgets.get("judge_name")
judge_model = dbutils.widgets.get("judge_model")
embedding_model = dbutils.widgets.get("embedding_model")

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
judge_name = dbutils.widgets.get("judge_name")
judge_model = dbutils.widgets.get("judge_model")
embedding_model = dbutils.widgets.get("embedding_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure environment for litellm/DSPy routing

# COMMAND ----------

import os
import mlflow

# Configure Databricks FMAPI routing for litellm/DSPy
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _ws_url.startswith("http"):
    _ws_url = f"https://{_ws_url}"

os.environ["DATABRICKS_API_KEY"] = _token
os.environ["DATABRICKS_API_BASE"] = f"{_ws_url}/serving-endpoints"
os.environ["DATABRICKS_HOST"] = _ws_url
os.environ["DATABRICKS_TOKEN"] = _token

# Model URIs for litellm routing (databricks:/<endpoint-name>)
JUDGE_MODEL = f"databricks:/{judge_model}"
REFLECTION_MODEL = f"databricks:/{judge_model}"
EMBEDDING_MODEL = f"databricks:/{embedding_model}"

print(f"Judge model: {JUDGE_MODEL}")
print(f"Reflection model: {REFLECTION_MODEL}")
print(f"Embedding model: {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load evaluation traces with expert feedback

# COMMAND ----------

# Search ALL experiments for traces with human feedback
# Review App feedback may be in different experiments (e.g., DeployAgent, PostDeploymentEval)
# and uses the default assessment name "User feedback"
_user = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_NAME = f"/Users/{_user}/{agent_name}"
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id
print(f"Primary experiment: {EXPERIMENT_NAME} (ID: {EXPERIMENT_ID})")

# Collect traces from ALL relevant experiments
from mlflow import MlflowClient
_client = MlflowClient()
_all_experiments = _client.search_experiments(max_results=100)

all_traces = []
experiment_ids_searched = []
for exp in _all_experiments:
    if "/Trash" in exp.name:
        continue
    try:
        traces = mlflow.search_traces(
            locations=[exp.experiment_id],
            max_results=500,
            return_type="list",
        )
        if traces:
            all_traces.extend(traces)
            experiment_ids_searched.append(f"{exp.name}: {len(traces)}")
    except Exception:
        pass

print(f"\nSearched {len(experiment_ids_searched)} experiments, found {len(all_traces)} total traces")
for e in experiment_ids_searched:
    print(f"  {e}")

# Identify traces with HUMAN feedback specifically
traces_with_human_feedback = []
assessment_name_counts = {}
human_assessment_names = set()

for trace in all_traces:
    assessments = getattr(trace.info, "assessments", []) or []
    has_human = False
    for a in assessments:
        source_type = getattr(a.source, "source_type", "?") if hasattr(a, "source") else "?"
        key = f"{a.name}({source_type})"
        assessment_name_counts[key] = assessment_name_counts.get(key, 0) + 1
        if source_type == "HUMAN":
            has_human = True
            human_assessment_names.add(a.name)
    if has_human:
        traces_with_human_feedback.append(trace)

print(f"\nTraces with HUMAN assessments: {len(traces_with_human_feedback)}")
print(f"Human assessment names found: {human_assessment_names}")
print(f"\nFull assessment breakdown:")
for name, count in sorted(assessment_name_counts.items()):
    print(f"  {name}: {count}")

# Show sample human feedback
print(f"\nSample human feedback:")
_shown = 0
for trace in traces_with_human_feedback:
    for a in (getattr(trace.info, "assessments", []) or []):
        source_type = getattr(a.source, "source_type", "?") if hasattr(a, "source") else "?"
        if source_type == "HUMAN" and a.name != "expected_response":
            source_id = getattr(a.source, "source_id", "") if hasattr(a, "source") else ""
            val = getattr(a, "boolean_value", getattr(a, "numeric_value", getattr(a, "value", "?")))
            rationale = (getattr(a, "rationale", "") or "")[:100]
            print(f"  {trace.info.request_id[:30]}... name='{a.name}' value={val} by={source_id} reason={rationale}")
            _shown += 1
    if _shown >= 10:
        break

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create base judge and MemAlign optimizer

# COMMAND ----------

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer

# Detect which HUMAN assessment name has boolean feedback for alignment
# MemAlign requires traces with human assessments matching the judge name
# Review App uses "User feedback" by default; programmatic uses custom names
# Exclude "expected_response" (ground truth strings, not quality judgments)
HUMAN_FEEDBACK_NAME = None
for name in human_assessment_names:
    if name != "expected_response":
        HUMAN_FEEDBACK_NAME = name
        break

# Save the widget judge_name for consistent aligned judge naming across pipeline
# (e.g., "response_quality" → "response_quality_aligned" used by steps 06-09)
OUTPUT_JUDGE_NAME = judge_name  # From widget, default: "response_quality"

if HUMAN_FEEDBACK_NAME:
    print(f"\nUsing human feedback assessment: '{HUMAN_FEEDBACK_NAME}'")
    # Base judge name must match the assessment name for MemAlign to find traces
    base_judge_name = HUMAN_FEEDBACK_NAME
    # Filter to only traces with this specific human feedback
    traces_with_feedback = [
        t for t in traces_with_human_feedback
        if any(a.name == HUMAN_FEEDBACK_NAME and getattr(a.source, "source_type", "") == "HUMAN"
               for a in (getattr(t.info, "assessments", []) or []))
    ]
    print(f"Traces with '{HUMAN_FEEDBACK_NAME}' feedback: {len(traces_with_feedback)}")
else:
    print(f"\nNo boolean human feedback found (only 'expected_response' ground truth)")
    base_judge_name = judge_name
    traces_with_feedback = traces_with_human_feedback

# Determine feedback value type from the human assessments
_sample_value = None
for trace in all_traces:
    for a in (getattr(trace.info, "assessments", []) or []):
        if a.name == base_judge_name:
            _sample_value = getattr(a, "boolean_value", getattr(a, "numeric_value", getattr(a, "value", None)))
            break
    if _sample_value is not None:
        break

if isinstance(_sample_value, bool):
    feedback_type = bool
elif isinstance(_sample_value, (int, float)):
    feedback_type = float
else:
    # String-valued feedback (like expected_response) — use str
    feedback_type = str

print(f"Feedback value type: {feedback_type.__name__} (sample: {str(_sample_value)[:100]})")

# Create the base judge — name must match assessment name for MemAlign trace matching
base_judge = make_judge(
    name=base_judge_name,
    instructions=(
        "Evaluate the quality of the agent's response to a Databricks documentation question.\n\n"
        "Question: {{ inputs }}\n"
        "Response: {{ outputs }}\n\n"
        "Consider:\n"
        "- Accuracy: Is the response factually correct based on Databricks documentation?\n"
        "- Code quality: Does it include correct code snippets when relevant?\n"
        "- Source citation: Does it reference documentation sources?\n"
        "- Conciseness: Is it clear and actionable?\n"
        "- Completeness: Does it fully address the question?"
    ),
    feedback_value_type=feedback_type,
    model=JUDGE_MODEL,
)
print(f"Created base judge: '{base_judge_name}' (feedback_type={feedback_type.__name__})")
print(f"Aligned judge will be registered as: '{OUTPUT_JUDGE_NAME}_aligned'")

# Create MemAlign optimizer
# - reflection_lm: model for distilling guidelines from expert feedback
# - retrieval_k: number of similar examples for episodic memory retrieval
# - embedding_model: model for episodic memory embeddings
optimizer = MemAlignOptimizer(
    reflection_lm=REFLECTION_MODEL,
    retrieval_k=3,
    embedding_model=EMBEDDING_MODEL,
)

print(f"\nMemAlignOptimizer configured:")
print(f"  reflection_lm: {REFLECTION_MODEL}")
print(f"  retrieval_k: 3")
print(f"  embedding_model: {EMBEDDING_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Align the judge with expert feedback
# MAGIC
# MAGIC `base_judge.align(traces, optimizer)` uses MemAlign to:
# MAGIC - Distill **semantic memory** (generalizable guidelines) from expert annotations
# MAGIC - Build **episodic memory** (scored examples) for few-shot retrieval
# MAGIC - Produce an aligned judge that scores like the domain experts

# COMMAND ----------

# ── Pre-process: Convert string expected_response assessments to boolean feedback ──
# MemAlign requires boolean/numeric human assessments matching the judge name.
# If the HUMAN assessments are string-valued (ground truth answers from eval dataset),
# we generate boolean quality feedback by comparing agent output to expected response
# using the LLM judge, then log it as human feedback for alignment.

import requests as _requests

if feedback_type == str:
    print("Human assessments are string-valued (ground truth) — generating boolean quality feedback...")
    base_judge_name = OUTPUT_JUDGE_NAME  # Use widget judge_name for auto-generated feedback

    # Recreate judge with boolean type
    base_judge = make_judge(
        name=base_judge_name,
        instructions=(
            "Evaluate the quality of the agent's response to a Databricks documentation question.\n\n"
            "Question: {{ inputs }}\n"
            "Response: {{ outputs }}\n"
            "Expected answer: {{ expectations }}\n\n"
            "Is the response correct and helpful? Consider accuracy, completeness, and relevance."
        ),
        feedback_value_type=bool,
        model=JUDGE_MODEL,
    )
    print(f"Recreated judge '{base_judge_name}' with boolean feedback type")

    # Log boolean feedback on traces that have expected_response
    import json as _json
    logged_count = 0
    for trace in traces_with_feedback:
        # Get expected response from existing assessment
        expected = None
        for a in (getattr(trace.info, "assessments", []) or []):
            if a.name == "expected_response":
                expected = getattr(a, "value", getattr(a, "string_value", None))
                break
        if not expected:
            continue

        # Get agent's actual response from trace
        actual_response = ""
        try:
            resp_data = trace.data.response
            if isinstance(resp_data, str):
                resp_data = _json.loads(resp_data)
            if isinstance(resp_data, dict):
                msgs = resp_data.get("messages", resp_data.get("choices", []))
                if msgs:
                    if isinstance(msgs[0], dict):
                        actual_response = msgs[0].get("content", str(msgs[0]))
                    else:
                        actual_response = str(msgs[0])
        except Exception:
            continue

        if not actual_response:
            continue

        # Use LLM to judge: does the response match the expected answer?
        try:
            judge_prompt = (
                f"Does this response adequately answer the question based on the expected answer?\n\n"
                f"Expected: {expected[:500]}\n\nActual: {actual_response[:500]}\n\n"
                f"Answer with just 'yes' or 'no'."
            )
            from databricks.sdk import WorkspaceClient as _JW
            _jw = _JW()
            _jr = _jw.api_client.do(
                "POST", f"/serving-endpoints/{judge_model}/invocations",
                body={"messages": [{"role": "user", "content": judge_prompt}], "max_tokens": 10, "temperature": 0},
            )
            answer = _jr["choices"][0]["message"]["content"].strip().lower()
            is_good = answer.startswith("yes")

            # Log as human-equivalent feedback
            from mlflow.entities import AssessmentSource, AssessmentSourceType
            mlflow.log_feedback(
                trace_id=trace.info.request_id,
                name=base_judge_name,
                value=is_good,
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="auto_from_expected_response"),
                rationale=f"Auto-generated from expected_response comparison: {answer}",
            )
            logged_count += 1
        except Exception as e:
            continue

    print(f"Logged {logged_count} boolean feedback assessments as '{base_judge_name}'")

    # Re-fetch traces with the new feedback
    all_traces = mlflow.search_traces(locations=[EXPERIMENT_ID], max_results=500, return_type="list")
    traces_with_feedback = [
        t for t in all_traces
        if any(a.name == base_judge_name for a in (getattr(t.info, "assessments", []) or []))
    ]
    print(f"Traces with '{base_judge_name}' feedback: {len(traces_with_feedback)}")

print(f"\nAligning judge with {len(traces_with_feedback)} traces...")
aligned_judge = base_judge.align(traces=traces_with_feedback, optimizer=optimizer)
print(f"Alignment complete for judge: {aligned_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect semantic memory (distilled guidelines)

# COMMAND ----------

print("=" * 60)
print("SEMANTIC MEMORY (Distilled Guidelines)")
print("=" * 60)
if hasattr(aligned_judge, "_semantic_memory") and aligned_judge._semantic_memory:
    for i, guideline in enumerate(aligned_judge._semantic_memory, 1):
        print(f"\n{i}. {guideline.guideline_text}")
        if hasattr(guideline, "source_trace_ids") and guideline.source_trace_ids:
            ids = guideline.source_trace_ids
            print(f"   Source traces: {ids[:3]}..." if len(ids) > 3 else f"   Source traces: {ids}")
else:
    print("No semantic memory (guidelines) produced.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Inspect episodic memory (stored examples)

# COMMAND ----------

print("=" * 60)
print("EPISODIC MEMORY (Stored Examples)")
print("=" * 60)
if hasattr(aligned_judge, "_episodic_memory") and aligned_judge._episodic_memory:
    print(f"Total examples: {len(aligned_judge._episodic_memory)}")
    print(aligned_judge._episodic_memory)
else:
    print("No episodic memory produced.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Show aligned judge instructions

# COMMAND ----------

print("=" * 60)
print("ALIGNED JUDGE INSTRUCTIONS")
print("=" * 60)
print(aligned_judge.instructions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Register the aligned judge

# COMMAND ----------

from mlflow.genai.scorers import ScorerSamplingConfig

mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

ALIGNED_JUDGE_NAME = f"{OUTPUT_JUDGE_NAME}_aligned"
print(f"Will register as: {ALIGNED_JUDGE_NAME}")

# Register the aligned judge
try:
    registered_judge = aligned_judge.register(
        name=ALIGNED_JUDGE_NAME,
        experiment_id=EXPERIMENT_ID,
    )
    print(f"Registered aligned judge: {ALIGNED_JUDGE_NAME}")
except Exception as e:
    print(f"Register failed ({e}), trying update...")
    try:
        registered_judge = aligned_judge.update(
            experiment_id=EXPERIMENT_ID,
        )
        print(f"Updated aligned judge: {ALIGNED_JUDGE_NAME}")
    except Exception as e2:
        print(f"Update also failed: {e2}")
        print("Aligned judge available in memory — proceeding without persistence.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify aligned judge can be loaded

# COMMAND ----------

from mlflow.genai.scorers import get_scorer

try:
    reloaded_judge = get_scorer(name=ALIGNED_JUDGE_NAME)
    print(f"Reloaded aligned judge: {reloaded_judge.name}")
    if hasattr(reloaded_judge, "_semantic_memory") and reloaded_judge._semantic_memory:
        print(f"  Semantic memory: {len(reloaded_judge._semantic_memory)} guidelines")
    if hasattr(reloaded_judge, "_episodic_memory") and reloaded_judge._episodic_memory:
        print(f"  Episodic memory: {len(reloaded_judge._episodic_memory)} examples")
except Exception as e:
    print(f"Could not reload aligned judge from registry: {e}")
    print("Using in-memory aligned judge for subsequent steps.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="judge_alignment", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("load_traces", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(traces_with_feedback),
                  output_summary={"total_traces": len(all_traces), "traces_with_feedback": len(traces_with_feedback),
                                  "assessment_names": assessment_name_counts})

step = pipeline.start_step("align_judge", step_order=2, step_type="optimization")
semantic_count = len(aligned_judge._semantic_memory) if hasattr(aligned_judge, "_semantic_memory") and aligned_judge._semantic_memory else 0
episodic_count = len(aligned_judge._episodic_memory) if hasattr(aligned_judge, "_episodic_memory") and aligned_judge._episodic_memory else 0
pipeline.end_step(step, status="COMPLETED", output_summary={
    "optimizer": "MemAlign",
    "judge_name": ALIGNED_JUDGE_NAME,
    "semantic_guidelines": semantic_count,
    "episodic_examples": episodic_count,
})

pipeline.end(status="COMPLETED")

import json
dbutils.notebook.exit(json.dumps({
    "aligned_judge_name": ALIGNED_JUDGE_NAME,
    "traces_used": len(traces_with_feedback),
    "semantic_guidelines": semantic_count,
    "episodic_examples": episodic_count,
}))

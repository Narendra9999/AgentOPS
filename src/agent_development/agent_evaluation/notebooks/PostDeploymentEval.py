# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation Pipeline
# MAGIC Loads evaluation datasets into UC tables, runs MLflow evaluation + guardrail tests,
# MAGIC saves per-row results, and outputs CI/CD gate decision.

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("eval_golden_table", "eval_golden_dataset")
dbutils.widgets.text("eval_adversarial_table", "eval_adversarial_dataset")
dbutils.widgets.text("eval_results_table", "eval_results")
dbutils.widgets.text("chatbot_name", "agentops-docs-chatbot")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
eval_golden_table = dbutils.widgets.get("eval_golden_table")
eval_adversarial_table = dbutils.widgets.get("eval_adversarial_table")
eval_results_table = dbutils.widgets.get("eval_results_table")
chatbot_name = dbutils.widgets.get("chatbot_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies + start audit tracking

# COMMAND ----------

# Ensure MLflow 3.x with genai module (needed for mlflow.genai.evaluate)
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
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow>=3.1.0", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets after restart
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
environment = dbutils.widgets.get("environment")
eval_golden_table = dbutils.widgets.get("eval_golden_table")
eval_adversarial_table = dbutils.widgets.get("eval_adversarial_table")
eval_results_table = dbutils.widgets.get("eval_results_table")
chatbot_name = dbutils.widgets.get("chatbot_name")

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="post_deployment_eval", agent_name=agent_name, environment=environment,
    triggered_by="cicd", depends_on="smoke_test", spark=spark, dbutils=dbutils,
)
pipeline.start()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load evaluation datasets into UC tables

# COMMAND ----------

import pandas as pd
import json

# Create audit schema and tables if they don't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{audit_schema}")
from framework.audit.audit_logger import get_audit_ddls
for _tname, _ddl in get_audit_ddls(catalog, audit_schema).items():
    spark.sql(_ddl)
    print(f"Ensured table: {catalog}.{audit_schema}.{_tname}")

# Resolve path relative to this notebook
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]

# Golden dataset → UC table
golden_df = pd.read_json(f"/Workspace/{nb_root}/agent_evaluation/evaluation/golden_dataset.json")
spark.createDataFrame(golden_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_golden_table}")
print(f"Golden: {len(golden_df)} rows → {catalog}.{schema}.{eval_golden_table}")

# Adversarial dataset → UC table
adversarial_df = pd.read_json(f"/Workspace/{nb_root}/agent_evaluation/evaluation/adversarial_dataset.json")
spark.createDataFrame(adversarial_df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.{eval_adversarial_table}")
print(f"Adversarial: {len(adversarial_df)} rows → {catalog}.{schema}.{eval_adversarial_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Guardrail Evaluation

# COMMAND ----------

import yaml
from framework.guardrails.pre_llm import PreLLMGuardrails
from framework.evaluation.evaluation_pipeline import run_guardrail_evaluation

# Load guardrail config
with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
    agent_config = yaml.safe_load(f)

pre_guardrails = PreLLMGuardrails(agent_config.get("guardrails", {}).get("pre_llm", {}))
adversarial_pd = spark.table(f"{catalog}.{schema}.{eval_adversarial_table}").toPandas()

guardrail_results = run_guardrail_evaluation(
    agent=type("Agent", (), {"pre_llm_guardrails": pre_guardrails})(),
    adversarial_dataset=adversarial_pd,
)

print(f"Block accuracy:   {guardrail_results['block_accuracy']:.2%}")
print(f"Pass accuracy:    {guardrail_results['pass_accuracy']:.2%}")
print(f"Overall accuracy: {guardrail_results['overall_accuracy']:.2%}")
print(f"False positives:  {guardrail_results['false_positives']}")
print(f"False negatives:  {guardrail_results['false_negatives']}")

# Show incorrect results
for d in guardrail_results["details"]:
    if not d["correct"]:
        print(f"  WRONG: [{d['attack_type']}] should_block={d['should_block']}, was_blocked={d['was_blocked']}: {d['request']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run MLflow Quality Evaluation

# COMMAND ----------

from framework.evaluation.evaluation_pipeline import run_evaluation
import sys, os

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_dir)))
sys.path.insert(0, _project_root)

# Load scorers based on config.yaml evaluation settings
# When mode="all", scorer groups run in PARALLEL with MLflow trace spans per group
import yaml
_agent_dir = os.path.join(_project_root, "agent_development", "agent")
with open(os.path.join(_agent_dir, "config.yaml")) as f:
    _agent_config = yaml.safe_load(f)

from agent_development.agent_evaluation.evaluation.scorer_loader import (
    load_scorer_groups, get_thresholds,
)
eval_config = _agent_config.get("evaluation", {})
scorer_groups = load_scorer_groups(eval_config)
eval_thresholds = get_thresholds(eval_config)
scorer_mode = eval_config.get("scorer_mode", "builtin")
total_scorers = sum(len(s) for s in scorer_groups.values())
print(f"Scorer mode: {scorer_mode} → {total_scorers} scorers in {len(scorer_groups)} groups")
for gname, gscorers in scorer_groups.items():
    print(f"  {gname}: {len(gscorers)} scorers")

golden_pd = spark.table(f"{catalog}.{schema}.{eval_golden_table}").toPandas()

# Set env vars for endpoint query function
os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _ws_url.startswith("http"):
    _ws_url = f"https://{_ws_url}"
os.environ["DATABRICKS_HOST"] = _ws_url

# Find the deployed endpoint
from databricks.sdk import WorkspaceClient as _WC
_w = _WC()
_endpoint_name = None
try:
    _ep = _w.serving_endpoints.get(chatbot_name)
    if _ep.state and str(_ep.state.ready).endswith("READY"):
        _endpoint_name = chatbot_name
except Exception:
    pass

if not _endpoint_name:
    _match_prefix = f"agents_{catalog}-{schema}"
    for _ep in _w.serving_endpoints.list():
        if _ep.name.startswith(_match_prefix) and _ep.state and str(_ep.state.ready).endswith("READY"):
            _endpoint_name = _ep.name
            break
print(f"Evaluating endpoint: {_endpoint_name}")

# Run evaluation with parallel scorer groups
# Each group (builtin, llm_judge, domain) runs concurrently with its own MLflow trace span
eval_result = run_evaluation(
    eval_dataset=golden_pd,
    scorer_groups=scorer_groups,
    thresholds=eval_thresholds,
    model_endpoint=_endpoint_name,
)

print(f"\n=== Evaluation Metrics ===")
for k, v in sorted(eval_result.get("metrics", {}).items()):
    print(f"  {k}: {v}")

# Show parallel execution timing if available
if eval_result.get("group_results"):
    print(f"\n=== Parallel Execution ===")
    for gname, gresult in eval_result["group_results"].items():
        duration = gresult.get("duration_ms", 0)
        scorers = gresult.get("scorers", [])
        print(f"  {gname}: {duration}ms ({len(scorers)} scorers)")

print(f"\n=== Quality Gate ===")
print(f"Overall: {'PASSED' if eval_result['passed'] else 'FAILED'}")
for metric, gate in eval_result.get("gate_results", {}).items():
    status = "PASS" if gate["passed"] else "FAIL"
    print(f"  [{status}] {metric}: {gate['actual']:.3f} (threshold: {gate['threshold']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Save per-row results to UC audit table

# COMMAND ----------

from framework.evaluation.evaluation_pipeline import save_eval_results_to_table

evaluation_id = save_eval_results_to_table(
    spark=spark,
    eval_result=eval_result,
    catalog=catalog,
    audit_schema=audit_schema,
    agent_name=agent_name,
    environment=environment,
    results_table_name=eval_results_table,
)

print(f"Per-row results saved: evaluation_id={evaluation_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Preview per-row scores

# COMMAND ----------

display(spark.sql(f"""
    SELECT request,
           round(toxicity_score, 3) as toxicity,
           round(accuracy_score, 1) as accuracy,
           round(helpfulness_score, 1) as helpfulness,
           round(professionalism_score, 1) as professionalism,
           round(docs_relevance_score, 2) as docs_relevance,
           round(code_snippet_score, 2) as code_snippet,
           round(source_citation_score, 2) as citation,
           round(answer_completeness_score, 2) as completeness
    FROM {catalog}.{audit_schema}.{eval_results_table}
    WHERE evaluation_id = '{evaluation_id}'
    ORDER BY row_index
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Complete audit tracking

# COMMAND ----------

step = pipeline.start_step("guardrail_evaluation", step_order=1, step_type="post_eval", depends_on="smoke_test")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "block_accuracy": guardrail_results["block_accuracy"],
    "pass_accuracy": guardrail_results["pass_accuracy"],
    "overall_accuracy": guardrail_results["overall_accuracy"],
    "false_positives": guardrail_results["false_positives"],
    "false_negatives": guardrail_results["false_negatives"],
})

# Build full quality summary with all metrics
quality_summary = {
    "scorer_mode": scorer_mode,
    "scorer_groups": {k: len(v) for k, v in scorer_groups.items()},
    "gate_passed": eval_result["passed"],
    "all_metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in eval_result.get("metrics", {}).items()},
    "gate_results": eval_result.get("gate_results", {}),
}
if eval_result.get("group_results"):
    quality_summary["parallel_timing_ms"] = {
        gname: gresult.get("duration_ms", 0)
        for gname, gresult in eval_result["group_results"].items()
    }

step = pipeline.start_step("quality_evaluation", step_order=2, step_type="post_eval", depends_on="guardrail_evaluation")
pipeline.end_step(step, status="COMPLETED", records_processed=len(golden_pd), output_summary=quality_summary)

step = pipeline.start_step("save_eval_results", step_order=3, step_type="post_eval")
pipeline.end_step(step, status="COMPLETED", records_processed=len(golden_pd), output_summary={
    "evaluation_id": evaluation_id,
})

pipeline.end(status="COMPLETED" if eval_result["passed"] else "FAILED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CI/CD Gate Output

# COMMAND ----------

gate_passed = eval_result["passed"] and guardrail_results["overall_accuracy"] >= 0.95

print(f"\n{'='*50}")
print(f"  EVALUATION GATE: {'PASSED' if gate_passed else 'FAILED'}")
print(f"  Quality:   {'PASS' if eval_result['passed'] else 'FAIL'}")
print(f"  Guardrails: {guardrail_results['overall_accuracy']:.2%}")
print(f"{'='*50}")

dbutils.notebook.exit(json.dumps({
    "passed": bool(gate_passed),
    "quality_passed": bool(eval_result["passed"]),
    "guardrail_accuracy": float(guardrail_results["overall_accuracy"]),
    "evaluation_id": str(evaluation_id),
}))

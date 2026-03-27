# Databricks notebook source
# MAGIC %md
# MAGIC # Iterative Improvement — MemAlign + GEPA
# MAGIC Collect expert feedback, align the LLM judge, and optimize the system prompt.
# MAGIC
# MAGIC **Reference:** [Self-Optimizing Chatbot](https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks)
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Collect expert-labeled traces from MLflow
# MAGIC 2. Align LLM judge with expert preferences (MemAlign)
# MAGIC 3. Optimize system prompt using aligned judge (GEPA)
# MAGIC 4. Compare optimized prompt vs baseline

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")

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
# MAGIC ## 2. Collect expert feedback from traces

# COMMAND ----------

from framework.optimization.prompt_optimizer import IterativeOptimizer

optimizer = IterativeOptimizer(
    agent_name=agent_name,
    experiment_name=f"/Users/{spark.sql('SELECT current_user()').first()[0]}/{agent_name}",
)

feedback = optimizer.collect_expert_feedback(max_traces=200)
print(f"Expert-labeled traces: {len(feedback)}")

if feedback:
    for f in feedback[:3]:
        print(f"\n  Trace: {f['trace_id']}")
        print(f"  Input: {str(f['input'])[:100]}...")
        print(f"  Assessments: {len(f['assessments'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build alignment dataset

# COMMAND ----------

alignment_data = optimizer.build_alignment_dataset(feedback)
print(f"Alignment dataset: {len(alignment_data)} rows")

if not alignment_data.empty:
    display(spark.createDataFrame(alignment_data).limit(10))
else:
    print("No expert labels found. Label traces in the MLflow UI first:")
    print(f"  1. Open MLflow experiment: /Users/.../{{agent_name}}")
    print(f"  2. Click on a trace")
    print(f"  3. Add assessment (thumbs up/down + rationale)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Align LLM judge with expert preferences (MemAlign)

# COMMAND ----------

alignment_result = optimizer.align_judge(alignment_data)
print(f"Judge alignment: {alignment_result['status']}")
if alignment_result.get("num_labels"):
    print(f"  Labels used: {alignment_result['num_labels']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load evaluation dataset

# COMMAND ----------

golden_table = f"{catalog}.{schema}.eval_golden_dataset"
eval_dataset = spark.table(golden_table).toPandas()
print(f"Evaluation dataset: {len(eval_dataset)} rows from {golden_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Optimize system prompt (GEPA)

# COMMAND ----------

optimization_result = optimizer.optimize_prompt(
    current_prompt=current_prompt,
    eval_dataset=eval_dataset,
    guidelines=[
        "Response should be accurate and based on Databricks documentation",
        "Response should include code snippets for coding questions",
        "Response should cite source URLs from the documentation",
        "Response should be concise and actionable",
        "Response should include appropriate caveats when needed",
    ],
)

print(f"Optimization: {optimization_result['status']}")
print(f"Baseline score: {optimization_result.get('baseline_score')}")
if optimization_result.get("optimized_score"):
    improvement = optimization_result["optimized_score"] - optimization_result["baseline_score"]
    print(f"Optimized score: {optimization_result['optimized_score']}")
    print(f"Improvement: {improvement:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Review optimized prompt (if available)

# COMMAND ----------

if optimization_result.get("optimized_prompt"):
    print("=== Optimized System Prompt ===")
    print(optimization_result["optimized_prompt"])
    print("\n=== Next Steps ===")
    print("1. Review the optimized prompt above")
    print("2. If satisfied, update config.yaml with the new prompt")
    print("3. Re-register the agent (Agent.py notebook)")
    print("4. Run evaluation to verify improvement")
else:
    print("No optimized prompt generated yet.")
    print("This requires the GEPA API (mlflow.genai.optimize_prompts).")
    print("\nTo enable:")
    print("  1. Ensure MLflow 2.16+ is installed")
    print("  2. Have enough expert labels (10+ recommended)")
    print("  3. The aligned judge will drive optimization")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log to audit

# COMMAND ----------

import json
from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="iterative_improvement", agent_name=agent_name, environment="dev",
    triggered_by="manual",
)
pipeline.start()

step = pipeline.start_step("collect_feedback", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(feedback),
                  output_summary={"labeled_traces": len(feedback)})

step = pipeline.start_step("align_judge", step_order=2, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", output_summary=alignment_result)

step = pipeline.start_step("optimize_prompt", step_order=3, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "baseline_score": optimization_result.get("baseline_score"),
    "optimized_score": optimization_result.get("optimized_score"),
})

pipeline.end(status="COMPLETED")
print("Optimization cycle logged to audit tables")

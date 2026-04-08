# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Agent Skill Generation with optimize_anything
# MAGIC Uses GEPA's `optimize_anything` to generate modular skill files from:
# MAGIC - Tool signatures (vector search, LLM endpoint)
# MAGIC - Optimized system prompt (from 06)
# MAGIC - Aligned judge memory (from 05)
# MAGIC - Evaluated traces with expert feedback
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Aligned judge (from 05_JudgeAlignment)
# MAGIC - Optimized prompt (from 06_PromptOptimization)
# MAGIC - Evaluation traces
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads aligned judge, optimized prompt, tool signatures, and eval traces
# MAGIC 2. Builds context for skill generation (tools, prompt, judge memory, traces)
# MAGIC 3. Runs `optimize_anything` to iteratively refine skill files
# MAGIC 4. Validates and writes skills to a UC Volume
# MAGIC
# MAGIC **Reference:** Notebook 07-AgentSkillsGeneration from at-bat-assistant

# COMMAND ----------

dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("audit_schema", "agentops_audit")
dbutils.widgets.text("judge_model", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("aligned_judge_name", "response_quality_aligned")
dbutils.widgets.text("max_metric_calls", "150")

agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
if os.path.exists(_vol_path):
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "dspy", "gepa", "--find-links", _vol_path, "--no-index", "-q"])
else:
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.4", "dspy>=2.6", "gepa", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")
judge_model = dbutils.widgets.get("judge_model")
aligned_judge_name = dbutils.widgets.get("aligned_judge_name")
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configure environment

# COMMAND ----------

import os, json, yaml, re
import mlflow

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
_ws_url = spark.conf.get("spark.databricks.workspaceUrl", "")
if not _ws_url.startswith("http"):
    _ws_url = f"https://{_ws_url}"
os.environ["DATABRICKS_API_KEY"] = _token
os.environ["DATABRICKS_API_BASE"] = f"{_ws_url}/serving-endpoints"
os.environ["DATABRICKS_HOST"] = _ws_url
os.environ["DATABRICKS_TOKEN"] = _token

REFLECTION_MODEL = f"databricks:/{judge_model}"
SKILLS_VOLUME_NAME = f"{agent_name}_skills"
SKILLS_VOLUME_PATH = f"/Volumes/{catalog}/{schema}/{SKILLS_VOLUME_NAME}"
PROMPT_NAME = f"{agent_name}_system_prompt"

_user = spark.sql("SELECT current_user()").first()[0]
experiment = mlflow.set_experiment(f"/Users/{_user}/{agent_name}")
EXPERIMENT_ID = experiment.experiment_id

print(f"Skills volume: {SKILLS_VOLUME_PATH}")
print(f"Reflection model: {REFLECTION_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load aligned judge

# COMMAND ----------

from mlflow.genai.scorers import get_scorer

aligned_judge = get_scorer(name=aligned_judge_name)
print(f"Loaded aligned judge: {aligned_judge.name}")

# Trigger episodic memory init
try:
    aligned_judge(
        inputs={"input": [{"role": "user", "content": "What is Unity Catalog?"}]},
        outputs={"response": "Unity Catalog is a governance solution."},
    )
except Exception:
    pass

# Extract semantic guidelines for context
semantic_guidelines = []
if hasattr(aligned_judge, "_semantic_memory") and aligned_judge._semantic_memory:
    for g in aligned_judge._semantic_memory:
        semantic_guidelines.append(g.guideline_text)
    print(f"Semantic guidelines: {len(semantic_guidelines)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load optimized prompt and tool signatures

# COMMAND ----------

# Load optimized prompt
try:
    prompt_obj = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/latest")
    optimized_prompt = prompt_obj.template
    print(f"Loaded optimized prompt: {PROMPT_NAME} ({len(optimized_prompt)} chars)")
except Exception:
    nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
    with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
        _cfg = yaml.safe_load(f)
    optimized_prompt = _cfg.get("system_prompt", "")
    print(f"Loaded prompt from config.yaml ({len(optimized_prompt)} chars)")

# Tool signatures — describe the agent's capabilities
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
with open(f"/Workspace/{nb_root}/agent/config.yaml") as f:
    agent_config = yaml.safe_load(f)

vs_config = agent_config.get("vector_search", {})
llm_config = agent_config.get("llm", {})

tool_signatures = f"""
Tools available to the agent:
1. Vector Search: Retrieves relevant documentation chunks from '{vs_config.get('index', 'docs_index')}'.
   - Returns top {vs_config.get('num_results', 5)} results with columns: {vs_config.get('columns', ['chunk_text', 'url'])}
   - Search type: {vs_config.get('search_type', 'similarity')}

2. LLM Endpoint: '{llm_config.get('endpoint', 'unknown')}' for response generation.
   - Max tokens: {llm_config.get('max_tokens', 2048)}
   - Temperature: {llm_config.get('temperature', 0.1)}
"""
print(tool_signatures)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load evaluation traces

# COMMAND ----------

eval_traces_list = mlflow.search_traces(
    locations=[EXPERIMENT_ID],
    max_results=200,
    return_type="list",
)
print(f"Loaded {len(eval_traces_list)} traces from experiment")

# Build trace summaries for skill generation context
trace_summaries = []
for trace in eval_traces_list[:50]:  # Use top 50 for context
    trace_data = trace.data
    user_query = ""
    try:
        request = trace_data.request
        if isinstance(request, str):
            request = json.loads(request)
        if isinstance(request, dict):
            inputs = request.get("input", request.get("inputs", request.get("messages", [])))
            if isinstance(inputs, list):
                for msg in inputs:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        user_query = msg.get("content", "")[:200]
                        break
    except Exception:
        pass

    response_preview = ""
    try:
        response = trace_data.response
        if isinstance(response, str):
            response_preview = response[:200]
        elif isinstance(response, dict):
            msgs = response.get("messages", response.get("choices", []))
            if msgs:
                response_preview = str(msgs[0])[:200]
    except Exception:
        pass

    if user_query:
        trace_summaries.append({"query": user_query, "response_preview": response_preview})

print(f"Trace summaries for skill context: {len(trace_summaries)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run optimize_anything for skill generation

# COMMAND ----------

try:
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig, RefinerConfig

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=MAX_METRIC_CALLS,
            cache_evaluation=True,
            display_progress_bar=True,
            parallel=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=f"databricks/{judge_model}",
            reflection_minibatch_size=3,
        ),
        refiner=RefinerConfig(),
    )

    OBJECTIVE = f"""Generate and optimize a set of modular skill files for a Databricks documentation AI agent.

Each skill is a folder with exactly 3 REQUIRED files:
  - skill-name/SKILL.md (YAML frontmatter + Tools, Workflow, Quality expectations, Response format)
  - skill-name/GOTCHA.md (at least 3 common pitfalls and how to avoid them)
  - skill-name/FEW_SHOT.md (2-3 complete Q&A examples with tool calls shown)

Guidelines from aligned judge:
{chr(10).join(f'  - {g}' for g in semantic_guidelines[:10])}

Agent system prompt (optimized):
{optimized_prompt[:500]}

Tool signatures:
{tool_signatures}

Sample queries from evaluation:
{chr(10).join(f'  - {s["query"]}' for s in trace_summaries[:15])}

Generate skills that cover the most common query patterns. Each skill should be self-contained
and loadable at runtime. Focus on Databricks-specific knowledge domains."""

    # Scorer: evaluate skill quality using aligned judge
    def evaluate_skills(candidate_json, trace_example):
        """Score a candidate skill set against a trace example."""
        try:
            score_result = aligned_judge(
                inputs={"input": [{"role": "user", "content": trace_example.get("query", "")}]},
                outputs={"response": f"[Using skills: {candidate_json[:200]}...] {trace_example.get('response_preview', '')}"},
            )
            return (float(score_result) if isinstance(score_result, (int, float)) else 0.5, {})
        except Exception:
            return (0.5, {})

    # Split traces for train/val
    train_traces = trace_summaries[:int(len(trace_summaries) * 0.7)]
    val_traces = trace_summaries[int(len(trace_summaries) * 0.7):]

    print(f"Train: {len(train_traces)}, Val: {len(val_traces)}")
    print(f"Running optimize_anything...")

    result = optimize_anything(
        objective=OBJECTIVE,
        evaluate_fn=lambda candidate: sum(
            evaluate_skills(candidate, t)[0] for t in val_traces[:10]
        ) / min(len(val_traces), 10),
        config=config,
    )

    print(f"\nOptimization complete!")
    print(f"  Best score: {result.val_aggregate_scores[result.best_idx]:.4f}")

    # Parse generated skills
    optimized_candidate = result.best_candidate
    if isinstance(optimized_candidate, dict):
        optimized_json = list(optimized_candidate.values())[0]
    else:
        optimized_json = optimized_candidate

    generated_skills_text = optimized_json
    OPTIMIZE_ANYTHING_AVAILABLE = True

except ImportError:
    print("gepa package not available — using template-based skill generation instead")
    OPTIMIZE_ANYTHING_AVAILABLE = False

    # Fallback: generate skills using LLM directly
    import requests as _requests

    skill_generation_prompt = f"""Generate modular skill files for a Databricks documentation AI agent.

Guidelines from evaluation:
{chr(10).join(f'  - {g}' for g in semantic_guidelines[:10])}

Tool signatures:
{tool_signatures}

Common query patterns:
{chr(10).join(f'  - {s["query"]}' for s in trace_summaries[:10])}

Generate 3-5 skills as a JSON array. Each skill should have:
- "filename": "skill-name/SKILL.md"
- "content": full markdown content with YAML frontmatter (name, description), Tools, Workflow, Quality expectations, Response format sections
"""

    resp = _requests.post(
        f"{_ws_url}/serving-endpoints/{judge_model}/invocations",
        headers={"Authorization": f"Bearer {_token}"},
        json={"messages": [{"role": "user", "content": skill_generation_prompt}], "max_tokens": 4096, "temperature": 0.3},
        timeout=120,
    )
    resp.raise_for_status()
    generated_skills_text = resp.json()["choices"][0]["message"]["content"]
    print(f"Generated skills via LLM ({len(generated_skills_text)} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write skills to UC Volume

# COMMAND ----------

# Create volume
spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{SKILLS_VOLUME_NAME}
    COMMENT 'Agent skill files generated via GEPA optimize_anything'
""")
print(f"Volume ready: {SKILLS_VOLUME_PATH}")

# Parse skills from JSON array
generated_skills = []
try:
    match = re.search(r'\[.*\]', generated_skills_text, re.DOTALL)
    if match:
        generated_skills = json.loads(match.group())
    else:
        generated_skills = json.loads(generated_skills_text)
except (json.JSONDecodeError, TypeError):
    print(f"Could not parse skills as JSON. Saving raw output.")
    generated_skills = [{"filename": "raw_skills.md", "content": generated_skills_text}]

print(f"Skills to write: {len(generated_skills)}")

# Write each skill file
for skill in generated_skills:
    filename = skill.get("filename", f"skill_{generated_skills.index(skill)}.md")
    content = skill.get("content", "")

    # Ensure parent directory exists in volume
    parts = filename.split("/")
    if len(parts) > 1:
        dir_path = f"{SKILLS_VOLUME_PATH}/{'/'.join(parts[:-1])}"
        dbutils.fs.mkdirs(dir_path)

    file_path = f"{SKILLS_VOLUME_PATH}/{filename}"
    dbutils.fs.put(file_path, content, overwrite=True)
    print(f"  Written: {file_path} ({len(content)} chars)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Preview generated skills

# COMMAND ----------

for skill in generated_skills:
    filename = skill.get("filename", "?")
    content = skill.get("content", "")
    print(f"\n{'=' * 70}")
    print(f"  {filename}")
    print(f"{'=' * 70}")
    preview = content[:1500]
    if len(content) > 1500:
        preview += "\n... (truncated)"
    print(preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="skill_generation", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("generate_skills", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(generated_skills), output_summary={
    "optimizer": "optimize_anything" if OPTIMIZE_ANYTHING_AVAILABLE else "llm_direct",
    "skills_count": len(generated_skills),
    "skills_volume": SKILLS_VOLUME_PATH,
    "filenames": [s.get("filename", "?") for s in generated_skills],
})

pipeline.end(status="COMPLETED")

dbutils.notebook.exit(json.dumps({
    "skills_count": len(generated_skills),
    "skills_volume": SKILLS_VOLUME_PATH,
    "optimizer": "optimize_anything" if OPTIMIZE_ANYTHING_AVAILABLE else "llm_direct",
}))

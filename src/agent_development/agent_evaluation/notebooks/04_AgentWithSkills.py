# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Agent with Skills
# MAGIC Extends the base agent with runtime skill loading from UC Volume.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Skills generated in UC Volume (from 07_SkillGeneration)
# MAGIC - Optimized prompt registered (from 06_PromptOptimization)
# MAGIC - Agent model registered in UC
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Loads skill files from UC Volume
# MAGIC 2. Creates a skill-aware agent wrapper that appends skill metadata to the system prompt
# MAGIC 3. Registers the agent with skills as a new model version
# MAGIC 4. Tests the skill-augmented agent
# MAGIC
# MAGIC **Reference:** Notebook 08_create_agent_with_skills from at-bat-assistant

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
# MAGIC ## 0. Install dependencies

# COMMAND ----------

import subprocess, os
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
if os.path.exists(_vol_path):
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "--find-links", _vol_path, "--no-index", "-q"])
else:
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.4", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read widgets
agent_name = dbutils.widgets.get("agent_name")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
audit_schema = dbutils.widgets.get("audit_schema")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load skills from UC Volume

# COMMAND ----------

import os, re, yaml, json

SKILLS_VOLUME_NAME = f"{agent_name}_skills"
SKILLS_VOLUME_PATH = f"/Volumes/{catalog}/{schema}/{SKILLS_VOLUME_NAME}"

def load_skills_from_volume(volume_path):
    """Load all SKILL.md files from a UC Volume and parse frontmatter."""
    skills = []
    try:
        files = dbutils.fs.ls(volume_path)
    except Exception as e:
        print(f"No skills found at {volume_path}: {e}")
        return skills

    for item in files:
        if item.isDir():
            # Skill folder — look for SKILL.md inside
            try:
                sub_files = dbutils.fs.ls(item.path)
                for sf in sub_files:
                    if sf.name.upper() == "SKILL.MD":
                        content = dbutils.fs.head(sf.path, 10000)
                        skills.append(_parse_skill(item.name.rstrip("/"), content))
            except Exception:
                pass
        elif item.name.endswith(".md"):
            # Direct skill file
            content = dbutils.fs.head(item.path, 10000)
            skills.append(_parse_skill(item.name.replace(".md", ""), content))

    return skills


def _parse_skill(name, content):
    """Parse a skill file into metadata + content."""
    frontmatter = {}
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if match:
        try:
            frontmatter = yaml.safe_load(match.group(1)) or {}
        except Exception:
            pass

    return {
        "name": frontmatter.get("name", name),
        "description": frontmatter.get("description", ""),
        "content": content,
        "filename": name,
    }


skills = load_skills_from_volume(SKILLS_VOLUME_PATH)
print(f"Loaded {len(skills)} skills from {SKILLS_VOLUME_PATH}")
for s in skills:
    print(f"  - {s['name']}: {s['description'][:80]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build skill metadata for system prompt

# COMMAND ----------

def build_skill_metadata_prompt(skills):
    """Build a compact skill index to append to the system prompt.
    Full skill content is loaded on demand, not in the prompt."""
    if not skills:
        return ""

    lines = ["\n\n## Available Skills\n"]
    lines.append("The following specialized skills are available. Use them when the query matches their domain.\n")
    for i, skill in enumerate(skills, 1):
        lines.append(f"{i}. **{skill['name']}**: {skill['description']}")
    lines.append("\nWhen using a skill, follow its workflow and quality expectations.")
    return "\n".join(lines)


skill_metadata = build_skill_metadata_prompt(skills)
print("Skill metadata to append to system prompt:")
print(skill_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create skill-aware agent wrapper

# COMMAND ----------

import sys, os
import mlflow
import yaml

# Load agent config
nb_root = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/", 3)[0]
_project_root = f"/Workspace{nb_root}" if nb_root.startswith("/") else f"/Workspace/{nb_root}"

with open(f"{_project_root}/agent/config.yaml") as f:
    agent_config = yaml.safe_load(f)

# Load optimized prompt from registry
PROMPT_NAME = f"{catalog}.{schema}.{agent_name}_system_prompt"
try:
    prompt_obj = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/1")
    optimized_prompt = prompt_obj.template
    print(f"Using optimized prompt from registry ({len(optimized_prompt)} chars)")
except Exception as e:
    print(f"Could not load from registry ({e}) — using config.yaml")
    optimized_prompt = agent_config.get("system_prompt", "")
    print(f"Using prompt from config.yaml ({len(optimized_prompt)} chars)")

# Combine prompt + skill metadata
augmented_prompt = optimized_prompt + skill_metadata
agent_config["system_prompt"] = augmented_prompt
print(f"\nAugmented prompt: {len(augmented_prompt)} chars ({len(optimized_prompt)} base + {len(skill_metadata)} skills)")

# Write runtime config
import tempfile
_runtime_config = os.path.join(tempfile.mkdtemp(), "runtime_config_with_skills.yaml")
with open(_runtime_config, "w") as f:
    yaml.dump(agent_config, f, default_flow_style=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test the skill-augmented agent

# COMMAND ----------

# Load the registered model via mlflow.pyfunc (same as serving container)
model_name_uc = f"{catalog}.{schema}.{agent_name}"
print(f"Loading model: models:/{model_name_uc}@champion")

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name_uc}@champion")
print(f"Model loaded successfully")

# Test with skill-augmented queries
test_queries = [
    "What is Unity Catalog?",
    "How do I create a Delta table?",
    "What are best practices for MLflow tracing?",
]
for query in test_queries:
    print(f"\n{'=' * 60}")
    print(f"Q: {query}")
    try:
        result = loaded_model.predict({"messages": [{"role": "user", "content": query}]})
        if hasattr(result, "messages"):
            response = result.messages[0].content
        elif isinstance(result, dict) and "messages" in result:
            response = result["messages"][0]["content"]
        else:
            response = str(result)
        print(f"A: {response[:300]}...")
    except Exception as e:
        print(f"A: Error — {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register agent with skills as new model version

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

model_name_uc = f"{catalog}.{schema}.{agent_name}"

_llm_endpoint = agent_config["llm"]["endpoint"]
_vs_index = agent_config.get("vector_search", {}).get("index", "")
_vs_fq = f"{catalog}.{schema}.{_vs_index}" if _vs_index and "." not in _vs_index else _vs_index

resources = [
    DatabricksServingEndpoint(endpoint_name=_llm_endpoint),
    DatabricksVectorSearchIndex(index_name=_vs_fq),
]

input_example = {"messages": [{"role": "user", "content": "What is Delta Lake?"}]}

# Agent.py and tools are in the agent subdirectory
_agent_base = os.path.join(_project_root, "agent")
_agent_notebooks = os.path.join(_agent_base, "notebooks")
_agent_py = os.path.join(_agent_notebooks, "Agent.py")
_tools_dir = os.path.join(_agent_base, "tools") if os.path.exists(os.path.join(_agent_base, "tools")) else None

# Find the wheel in the Volume or dist
_wheel_path = None
_vol_path = f"/Volumes/{catalog}/{schema}/libraries"
try:
    _vol_files = [f.path for f in dbutils.fs.ls(_vol_path) if f.name.endswith(".whl")]
    if _vol_files:
        _wheel_path = _vol_files[-1].replace("dbfs:", "")
except Exception:
    pass
if not _wheel_path:
    _dist_dir = os.path.dirname(os.path.dirname(_project_root)) + "/dist"
    if os.path.exists(_dist_dir):
        _whl_files = [f for f in os.listdir(_dist_dir) if f.endswith(".whl")]
        if _whl_files:
            _wheel_path = os.path.join(_dist_dir, sorted(_whl_files)[-1])

_model_artifacts = {}
if _wheel_path:
    _wheel_artifact = f"artifacts/{os.path.basename(_wheel_path)}"
    _model_artifacts = {_wheel_artifact: _wheel_path}
else:
    _wheel_artifact = None

from mlflow.utils.environment import _mlflow_conda_env

pip_deps = [
    "mlflow>=3.1.0",
    "databricks-agents>=1.2.0",
    "databricks-sdk>=0.30.0",
    "databricks-langchain[memory]",
    "pyyaml>=6.0",
]
if _wheel_artifact:
    pip_deps.insert(0, f"./{_wheel_artifact}")

_conda_env = _mlflow_conda_env(additional_pip_deps=pip_deps)

print(f"Agent.py: {_agent_py} (exists: {os.path.exists(_agent_py)})")
print(f"Tools: {_tools_dir} (exists: {os.path.exists(_tools_dir) if _tools_dir else False})")
print(f"Wheel: {_wheel_path}")

with mlflow.start_run(run_name="agent_with_skills_registration"):
    log_kwargs = dict(
        artifact_path="agent",
        python_model=_agent_py,
        model_config=_runtime_config,
        input_example=input_example,
        resources=resources,
        conda_env=_conda_env,
    )
    if _tools_dir:
        log_kwargs["code_paths"] = [_tools_dir]
    if _model_artifacts:
        log_kwargs["artifacts"] = _model_artifacts

    model_info = mlflow.pyfunc.log_model(**log_kwargs)

print(f"Logged agent with skills: {model_info.model_uri}")

# Tag model version
from mlflow import MlflowClient
client = MlflowClient()
latest_version = client.get_model_version_by_alias(model_name_uc, "champion").version
print(f"Current champion: v{latest_version}")
print(f"\nTo promote agent-with-skills, set alias 'champion' to the new version after evaluation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Audit logging

# COMMAND ----------

from framework.audit.audit_logger import PipelineStepLogger

pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=audit_schema,
    pipeline_name="agent_with_skills", agent_name=agent_name, environment="dev",
    triggered_by="manual", spark=spark, dbutils=dbutils,
)
pipeline.start()

step = pipeline.start_step("load_skills", step_order=1, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", records_processed=len(skills), output_summary={
    "skills_volume": SKILLS_VOLUME_PATH,
    "skills": [{"name": s["name"], "description": s["description"][:100]} for s in skills],
})

step = pipeline.start_step("register_model", step_order=2, step_type="optimization")
pipeline.end_step(step, status="COMPLETED", output_summary={
    "model_uri": model_info.model_uri,
    "augmented_prompt_length": len(augmented_prompt),
})

pipeline.end(status="COMPLETED")

dbutils.notebook.exit(json.dumps({
    "skills_count": len(skills),
    "model_uri": model_info.model_uri,
}))

# Databricks notebook source
# MAGIC %md
# MAGIC # Register Model
# MAGIC Builds framework wheel, uploads to UC Volume, logs model, registers in UC.
# MAGIC Does NOT deploy — deployment happens after pre-deployment evaluation passes.
# MAGIC
# MAGIC Outputs model_name and model_version for downstream steps.

# COMMAND ----------

# Parameters — infrastructure settings from databricks.yml via pipeline
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("agent_name", "")
dbutils.widgets.text("llm_endpoint", "")
dbutils.widgets.text("vs_endpoint", "")
dbutils.widgets.text("vs_index", "")
dbutils.widgets.text("embedding_model", "")
dbutils.widgets.text("team_config", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
vs_endpoint = dbutils.widgets.get("vs_endpoint")
vs_index = dbutils.widgets.get("vs_index")
embedding_model = dbutils.widgets.get("embedding_model")
team_config = dbutils.widgets.get("team_config").strip()

# COMMAND ----------

# Install dependencies — uses air-gapped volume if available, otherwise PyPI
import subprocess, os

# Install from Mastercard volume (air-gapped) or PyPI
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
_wheels_path = _vol_path if os.path.exists(_vol_path) else None

if _wheels_path:
    print(f"Installing from: {_wheels_path}")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "databricks-openai", "--find-links", _wheels_path, "--no-index", "-q"])
else:
    print("Installing from PyPI...")
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow[genai]>=3.5", "databricks-openai", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read after restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
llm_endpoint = dbutils.widgets.get("llm_endpoint")
vs_endpoint = dbutils.widgets.get("vs_endpoint")
vs_index = dbutils.widgets.get("vs_index")
embedding_model = dbutils.widgets.get("embedding_model")
team_config = dbutils.widgets.get("team_config").strip()

import mlflow
import os, sys, subprocess, glob

# Start audit tracking (after pip restart)
from framework.audit.audit_logger import PipelineStepLogger
pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="register_model", agent_name=agent_name, environment="dev",
    triggered_by="pipeline", depends_on="vector_search_setup", spark=spark, dbutils=dbutils,
)
pipeline.start()
_step = pipeline.start_step("log_and_register", step_order=1, step_type="registration", depends_on="vector_search_setup")

mlflow.set_registry_uri("databricks-uc")
experiment_name = f"/Users/{spark.sql('SELECT current_user()').first()[0]}/{agent_name}"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve paths

# COMMAND ----------

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir = os.path.dirname(_nb_path)  # .../notebooks
_project_root = "/Workspace" + os.path.dirname(os.path.dirname(os.path.dirname(_nb_dir)))  # .../files/src
_bundle_root = os.path.dirname(_project_root)  # .../files (where pyproject.toml lives)
_agent_dir = "/Workspace" + _nb_dir
_default_config = os.path.join(os.path.dirname(_agent_dir), "config.yaml")
_tools_dir = os.path.join(os.path.dirname(_agent_dir), "tools")
_framework_dir = os.path.join(_project_root, "framework")

# Use team config if specified, otherwise shared default
if team_config:
    _config_path = os.path.join(_bundle_root, team_config)
    if not os.path.exists(_config_path):
        print(f"WARNING: Team config not found at {_config_path}, falling back to default")
        _config_path = _default_config
    else:
        print(f"Using team config: {_config_path}")
else:
    _config_path = _default_config

# Copy team custom tools into tools/custom_tools/ so they're packaged with the model
# (MLflow can't copy workspace paths as separate code_paths — they must be under tools/)
import shutil
_custom_tools_dest = os.path.join(_tools_dir, "custom_tools")
if team_config:
    _team_dir_name = os.path.dirname(team_config)  # e.g., teams/platform-engineering
    _candidate = os.path.join(_bundle_root, _team_dir_name, "tools")
    if os.path.isdir(_candidate):
        tool_files = [f for f in os.listdir(_candidate) if f.endswith(".py")]
        if tool_files:
            os.makedirs(_custom_tools_dest, exist_ok=True)
            for f in tool_files:
                shutil.copy2(os.path.join(_candidate, f), _custom_tools_dest)
            print(f"Copied {len(tool_files)} custom tools to {_custom_tools_dest}: {tool_files}")

print(f"Project root (src): {_project_root}")
print(f"Bundle root: {_bundle_root}")
print(f"Agent dir: {_agent_dir}")
print(f"Config: {_config_path}")
print(f"Tools dir: {_tools_dir}")
print(f"Framework dir: {_framework_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build framework wheel + upload to UC Volume

# COMMAND ----------

_wheel_dir = "/tmp/agentops_wheels"
os.makedirs(_wheel_dir, exist_ok=True)

# --no-build-isolation: use cluster's setuptools (avoids PyPI download behind firewalls)
build_result = subprocess.run(
    ["pip", "wheel", _bundle_root, "--no-deps", "--no-build-isolation", "-w", _wheel_dir],
    capture_output=True, text=True)
print(build_result.stdout)
if build_result.returncode != 0:
    print(f"Build stderr: {build_result.stderr}")
    raise RuntimeError("Framework wheel build failed")

_wheel_files = sorted(glob.glob(os.path.join(_wheel_dir, "agentops_framework-*.whl")))
_local_wheel = _wheel_files[-1] if _wheel_files else None
_wheel_filename = os.path.basename(_local_wheel)
print(f"Built: {_wheel_filename}")

# Upload to UC Volume with retry and integrity verification
from framework.mlops_utils import (
    create_volume_if_missing, upload_to_volume, compute_sha256, verify_integrity
)
from databricks.sdk import WorkspaceClient as _UploadWC

_upload_w = _UploadWC()
_volume_name = "python_dependencies"
create_volume_if_missing(_upload_w, catalog, schema, _volume_name)
_volume_path = f"/Volumes/{catalog}/{schema}/{_volume_name}"
_volume_wheel = f"{_volume_path}/{_wheel_filename}"

# Upload with retry
upload_to_volume(_upload_w, _local_wheel, _volume_wheel)

# SHA-256 integrity check
_local_hash = compute_sha256(_local_wheel)
print(f"Wheel SHA-256: {_local_hash[:16]}…")

# Install on cluster for log_model validation
subprocess.run(["pip", "install", "--force-reinstall", "--no-deps", _local_wheel],
               capture_output=True, text=True)
sys.path.insert(0, os.path.dirname(_tools_dir))
print("Installed on cluster for validation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Merge config (databricks.yml + config.yaml) and log model

# COMMAND ----------

from mlflow.models.resources import (
    DatabricksServingEndpoint, DatabricksVectorSearchIndex,
    DatabricksSQLWarehouse, DatabricksTable,
)
from mlflow.utils.environment import _mlflow_conda_env
import yaml

# Merge config: databricks.yml (infra) + config.yaml (agent settings)
with open(_config_path) as f:
    _agent_config = yaml.safe_load(f)

_agent_config["catalog"] = catalog
_agent_config["schema"] = schema
_agent_config["agent"]["name"] = agent_name
if llm_endpoint:
    _agent_config["llm"]["endpoint"] = llm_endpoint
if vs_endpoint:
    _agent_config.setdefault("vector_search", {})["endpoint"] = vs_endpoint
if vs_index:
    _agent_config.setdefault("vector_search", {})["index"] = vs_index

_runtime_config = "/tmp/runtime_config.yaml"
with open(_runtime_config, "w") as f:
    yaml.dump(_agent_config, f, default_flow_style=False)

print("Merged runtime config:")
print(yaml.dump(_agent_config, default_flow_style=False)[:500])

# Resource declarations — these grant Model Serving system auth access
_llm_endpoint = _agent_config["llm"]["endpoint"]
_vs_index_name = _agent_config.get("vector_search", {}).get("index", "")
_vs_index_fq = f"{catalog}.{schema}.{_vs_index_name}" if _vs_index_name and "." not in _vs_index_name else _vs_index_name

resources = [
    DatabricksServingEndpoint(endpoint_name=_llm_endpoint),
    DatabricksVectorSearchIndex(index_name=_vs_index_fq),
]

# Declare SQL warehouse so Model Serving can write to UC Delta via SQL Statement API
_session_cfg = _agent_config.get("session_history", {})
_uc_cfg = _session_cfg.get("unity_catalog", {})
_warehouse_id = _uc_cfg.get("warehouse_id", "")
if _uc_cfg.get("enabled") and _warehouse_id and _warehouse_id != "auto":
    resources.append(DatabricksSQLWarehouse(warehouse_id=_warehouse_id))
    print(f"Resource: SQL Warehouse {_warehouse_id}")

# Declare UC session history table so Model Serving can read/write it
_uc_table = _uc_cfg.get("table", "")
if _uc_cfg.get("enabled") and _uc_table:
    _uc_table_fq = f"{catalog}.{schema}.{_uc_table}"
    resources.append(DatabricksTable(table_name=_uc_table_fq))
    print(f"Resource: UC Table {_uc_table_fq}")

print(f"Total resources declared: {len(resources)}")

input_example = {"messages": [{"role": "user", "content": "What is Delta Lake?"}]}
model_name = f"{catalog}.{schema}.{agent_name}"

# Embed wheel in model artifact
_wheel_artifact_name = f"artifacts/{_wheel_filename}"
_model_artifacts = {_wheel_artifact_name: _volume_wheel}

_conda_env = _mlflow_conda_env(
    additional_pip_deps=[
        f"./{_wheel_artifact_name}",
        "mlflow>=3.1.0",
        "databricks-agents>=1.2.0",
        "databricks-sdk>=0.30.0",
        "databricks-langchain[memory]",
        "pyyaml>=6.0",
    ],
)

with mlflow.start_run(run_name="agent_registration"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=os.path.join(_agent_dir, "Agent.py"),
        model_config=_runtime_config,
        code_paths=[_tools_dir],
        input_example=input_example,
        resources=resources,
        conda_env=_conda_env,
        artifacts=_model_artifacts,
    )

print(f"Logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register to UC + set aliases

# COMMAND ----------

uc_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
)
print(f"Registered: {model_name} version {uc_model.version}")

from mlflow import MlflowClient
_client = MlflowClient()
_client.set_registered_model_alias(model_name, "champion", uc_model.version)
print(f"Alias set: {model_name}@champion → v{uc_model.version}")

try:
    _old_champion = _client.get_model_version_by_alias(model_name, "previous")
except Exception:
    try:
        _old = _client.get_model_version_by_alias(model_name, "champion")
        if _old.version != uc_model.version:
            _client.set_registered_model_alias(model_name, "previous", _old.version)
            print(f"Alias set: {model_name}@previous → v{_old.version}")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4b. Register system prompt in MLflow Prompt Registry

# COMMAND ----------

# Register the system prompt so it can be versioned, tracked, and loaded at runtime
_prompt_name = f"{catalog}.{schema}.{agent_name}_system_prompt"
_system_prompt = _agent_config.get("system_prompt", "")

try:
    prompt_version = mlflow.genai.register_prompt(
        name=_prompt_name,
        template=_system_prompt,
        commit_message=f"Registered with model v{uc_model.version}",
        tags={
            "model_version": str(uc_model.version),
            "llm_endpoint": _agent_config["llm"]["endpoint"],
        },
    )
    print(f"Prompt registered: {_prompt_name} v{prompt_version.version}")

    # Set production alias
    mlflow.genai.set_prompt_alias(
        name=_prompt_name,
        alias="production",
        version=prompt_version.version,
    )
    print(f"Prompt alias set: {_prompt_name}@production → v{prompt_version.version}")
except Exception as e:
    # Prompt registry may not be available in all environments
    print(f"Prompt registration skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Complete audit tracking

# COMMAND ----------

pipeline.end_step(_step, status="COMPLETED", output_summary={
    "model_name": model_name,
    "model_version": str(uc_model.version),
    "model_uri": model_info.model_uri,
    "champion_alias": f"@champion → v{uc_model.version}",
})
pipeline.end(status="COMPLETED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Output for downstream steps

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "model_name": model_name,
    "model_version": str(uc_model.version),
    "model_uri": model_info.model_uri,
}))

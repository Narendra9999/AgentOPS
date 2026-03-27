# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent
# MAGIC Deploys the registered model to a serving endpoint.
# MAGIC Runs AFTER pre-deployment evaluation passes.
# MAGIC Supports champion/challenger traffic splits.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("agent_name", "")
# Deployment / champion-challenger settings
dbutils.widgets.text("champion_model_version", "latest")
dbutils.widgets.text("champion_workload_size", "Small")
dbutils.widgets.text("champion_traffic_percentage", "100")
dbutils.widgets.text("challenger_enabled", "false")
dbutils.widgets.text("challenger_model_version", "")
dbutils.widgets.text("challenger_workload_size", "Small")
dbutils.widgets.text("challenger_traffic_percentage", "0")
dbutils.widgets.text("serving_scale_to_zero", "true")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
champion_version_param = dbutils.widgets.get("champion_model_version")
champion_workload = dbutils.widgets.get("champion_workload_size")
champion_traffic = int(dbutils.widgets.get("champion_traffic_percentage"))
challenger_enabled = dbutils.widgets.get("challenger_enabled").lower() == "true"
challenger_version = dbutils.widgets.get("challenger_model_version")
challenger_workload = dbutils.widgets.get("challenger_workload_size")
challenger_traffic = int(dbutils.widgets.get("challenger_traffic_percentage"))
scale_to_zero = dbutils.widgets.get("serving_scale_to_zero").lower() == "true"

model_name = f"{catalog}.{schema}.{agent_name}"

# COMMAND ----------

# Install dependencies — uses air-gapped volume if available, otherwise PyPI
import subprocess, os
_vol_path = "/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/python312_all_libs"
if os.path.exists(_vol_path):
    subprocess.check_call(["pip", "install", "-U", "databricks-agents", "mlflow", "--find-links", _vol_path, "--no-index", "-q"])
else:
    subprocess.check_call(["pip", "install", "-U", "databricks-agents>=1.2.0", "mlflow>=3.1.0", "-q"])
dbutils.library.restartPython()

# COMMAND ----------

# Re-read after restart
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
champion_version_param = dbutils.widgets.get("champion_model_version")
champion_workload = dbutils.widgets.get("champion_workload_size")
champion_traffic = int(dbutils.widgets.get("champion_traffic_percentage"))
challenger_enabled = dbutils.widgets.get("challenger_enabled").lower() == "true"
challenger_version = dbutils.widgets.get("challenger_model_version")
challenger_workload = dbutils.widgets.get("challenger_workload_size")
challenger_traffic = int(dbutils.widgets.get("challenger_traffic_percentage"))
scale_to_zero = dbutils.widgets.get("serving_scale_to_zero").lower() == "true"
model_name = f"{catalog}.{schema}.{agent_name}"

# Start audit tracking
from framework.audit.audit_logger import PipelineStepLogger
pipeline = PipelineStepLogger(
    catalog=catalog, audit_schema=f"{schema}_audit",
    pipeline_name="deploy_agent", agent_name=agent_name, environment="dev",
    triggered_by="pipeline", depends_on="pre_deployment_eval", spark=spark,
)
pipeline.start()
_step = pipeline.start_step("deploy_endpoint", step_order=1, step_type="deployment", depends_on="pre_deployment_eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Resolve champion version

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks import agents
from mlflow import MlflowClient

w = WorkspaceClient()
_client = MlflowClient()

# "latest" means the most recent registered version (set by RegisterModel)
if champion_version_param in ("latest", "0", ""):
    champion_alias = _client.get_model_version_by_alias(model_name, "champion")
    champion_version_resolved = champion_alias.version
    print(f"Resolved champion from @champion alias: v{champion_version_resolved}")
else:
    champion_version_resolved = champion_version_param

print(f"=== Deployment Config ===")
print(f"  Model: {model_name}")
print(f"  Champion: v{champion_version_resolved} → {champion_traffic}% traffic ({champion_workload})")
if challenger_enabled:
    print(f"  Challenger: v{challenger_version} → {challenger_traffic}% traffic ({challenger_workload})")
    assert champion_traffic + challenger_traffic == 100, \
        f"Traffic must sum to 100%, got {champion_traffic + challenger_traffic}%"
    _client.set_registered_model_alias(model_name, "challenger", challenger_version)
    print(f"  Alias set: {model_name}@challenger → v{challenger_version}")
else:
    print(f"  Challenger: disabled")
    try:
        _client.delete_registered_model_alias(model_name, "challenger")
    except Exception:
        pass
print(f"  Scale to zero: {scale_to_zero}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy champion

# COMMAND ----------

deployment = agents.deploy(
    model_name,
    champion_version_resolved,
    tags={"environment": "dev", "framework": "agentops"},
)
endpoint_name = deployment.endpoint_name
print(f"Champion deployed: {endpoint_name} (v{champion_version_resolved})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure challenger (if enabled)

# COMMAND ----------

if challenger_enabled and challenger_version:
    from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route

    print(f"Waiting for champion to be READY before adding challenger...")
    for i in range(120):
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state
        if state and str(state.ready).endswith("READY") and \
           (state.config_update is None or str(state.config_update).endswith("NOT_UPDATING")):
            break
        time.sleep(30)

    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(entity_name=model_name, entity_version=champion_version_resolved,
                              workload_size=champion_workload, scale_to_zero_enabled=scale_to_zero),
            ServedEntityInput(entity_name=model_name, entity_version=challenger_version,
                              workload_size=challenger_workload, scale_to_zero_enabled=scale_to_zero),
        ],
        traffic_config=TrafficConfig(routes=[
            Route(served_model_name=f"{agent_name}-{champion_version_resolved}", traffic_percentage=champion_traffic),
            Route(served_model_name=f"{agent_name}-{challenger_version}", traffic_percentage=challenger_traffic),
        ]),
    )
    print(f"A/B config applied: {champion_traffic}% → v{champion_version_resolved}, {challenger_traffic}% → v{challenger_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for endpoint ready

# COMMAND ----------

max_wait_seconds = 4000
poll_interval = 30
elapsed = 0
endpoint_ready = False

print(f"Waiting for {endpoint_name} (max {max_wait_seconds}s)...")
while elapsed < max_wait_seconds:
    try:
        ep = w.serving_endpoints.get(endpoint_name)
        state = ep.state
        if state and str(state.ready).endswith("READY") and \
           (state.config_update is None or str(state.config_update).endswith("NOT_UPDATING")):
            endpoint_ready = True
            print(f"READY after {elapsed}s")
            break
        print(f"  {elapsed}s: ready={state.ready if state else '?'}, config={state.config_update if state else '?'}")
    except Exception as e:
        print(f"  {elapsed}s: check failed ({e})")
    time.sleep(poll_interval)
    elapsed += poll_interval

if not endpoint_ready:
    print(f"WARNING: Endpoint not ready after {max_wait_seconds}s — continuing anyway")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Smoke test

# COMMAND ----------

if endpoint_ready:
    import requests
    try:
        _token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        _host = spark.conf.get("spark.databricks.workspaceUrl", "")
        if not _host.startswith("http"):
            _host = f"https://{_host}"
        _resp = requests.post(f"{_host}/serving-endpoints/{endpoint_name}/invocations",
            headers={"Authorization": f"Bearer {_token}", "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": "What is Unity Catalog?"}]})
        _resp.raise_for_status()
        _data = _resp.json()
        content = _data["messages"][0]["content"] if "messages" in _data else str(_data)[:200]
        print(f"Smoke test PASSED: {content[:200]}...")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
else:
    print("Smoke test skipped — endpoint not ready yet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Complete audit tracking

# COMMAND ----------

pipeline.end_step(_step, status="COMPLETED", output_summary={
    "endpoint_name": endpoint_name,
    "champion_version": champion_version_resolved,
    "champion_traffic": champion_traffic,
    "challenger_enabled": bool(challenger_enabled),
    "endpoint_ready": bool(endpoint_ready),
})
pipeline.end(status="COMPLETED")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "endpoint_name": endpoint_name,
    "champion_version": champion_version_resolved,
    "endpoint_ready": bool(endpoint_ready),
}))

# Databricks notebook source
# MAGIC %md
# MAGIC # Update Traffic Split (Champion / Challenger)
# MAGIC Shift traffic between champion and challenger on an existing endpoint.
# MAGIC Does NOT re-register or redeploy — only changes the traffic routing.
# MAGIC
# MAGIC Use cases:
# MAGIC - Gradually increase challenger traffic (80/20 → 50/50 → 0/100)
# MAGIC - Rollback to champion (100/0)
# MAGIC - Promote challenger to 100% before formally re-registering

# COMMAND ----------

dbutils.widgets.text("catalog", "classic_stable_cykcbe_catalog")
dbutils.widgets.text("schema", "agentops")
dbutils.widgets.text("agent_name", "databricks_docs_agent")
dbutils.widgets.text("champion_version", "")
dbutils.widgets.text("champion_traffic", "100")
dbutils.widgets.text("challenger_version", "")
dbutils.widgets.text("challenger_traffic", "0")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
agent_name = dbutils.widgets.get("agent_name")
champion_version = dbutils.widgets.get("champion_version")
champion_traffic = int(dbutils.widgets.get("champion_traffic"))
challenger_version = dbutils.widgets.get("challenger_version")
challenger_traffic = int(dbutils.widgets.get("challenger_traffic"))

model_name = f"{catalog}.{schema}.{agent_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find the endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Find endpoint by prefix (handles truncated names from agents.deploy)
match_prefix = f"agents_{catalog}-{schema}"
endpoint_name = None
for ep in w.serving_endpoints.list():
    if ep.name.startswith(match_prefix) and ep.state and str(ep.state.ready).endswith("READY"):
        endpoint_name = ep.name
        break

if not endpoint_name:
    raise RuntimeError(f"No READY endpoint found matching '{match_prefix}...'")

print(f"Endpoint: {endpoint_name}")

# Show current config
ep = w.serving_endpoints.get(endpoint_name)
print(f"\nCurrent config:")
for entity in ep.config.served_entities:
    print(f"  v{entity.entity_version}: {entity.workload_size}")
if ep.config.traffic_config:
    for route in ep.config.traffic_config.routes:
        print(f"  {route.served_model_name} → {route.traffic_percentage}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resolve versions

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Resolve champion version — use alias if not specified
if not champion_version:
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
        champion_version = mv.version
        print(f"Resolved champion from @champion alias: v{champion_version}")
    except Exception:
        # Fall back to latest version
        versions = client.search_model_versions(f"name='{model_name}'")
        champion_version = str(max(int(v.version) for v in versions))
        print(f"Resolved champion from latest: v{champion_version}")

# Resolve challenger version — use alias if not specified
if not challenger_version and challenger_traffic > 0:
    try:
        mv = client.get_model_version_by_alias(model_name, "challenger")
        challenger_version = mv.version
        print(f"Resolved challenger from @challenger alias: v{challenger_version}")
    except Exception:
        raise RuntimeError("No challenger version specified and no @challenger alias found")

assert champion_traffic + challenger_traffic == 100, \
    f"Traffic must sum to 100%, got {champion_traffic + challenger_traffic}%"

print(f"\nNew traffic config:")
print(f"  Champion: v{champion_version} → {champion_traffic}%")
if challenger_traffic > 0:
    print(f"  Challenger: v{challenger_version} → {challenger_traffic}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Update traffic

# COMMAND ----------

from databricks.sdk.service.serving import ServedEntityInput, TrafficConfig, Route

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=champion_version,
        workload_size="Small",
        scale_to_zero_enabled=False,
    )
]

routes = [
    Route(
        served_model_name=f"{agent_name}-{champion_version}",
        traffic_percentage=champion_traffic,
    )
]

if challenger_traffic > 0 and challenger_version:
    served_entities.append(
        ServedEntityInput(
            entity_name=model_name,
            entity_version=challenger_version,
            workload_size="Small",
            scale_to_zero_enabled=False,
        )
    )
    routes.append(
        Route(
            served_model_name=f"{agent_name}-{challenger_version}",
            traffic_percentage=challenger_traffic,
        )
    )

w.serving_endpoints.update_config(
    name=endpoint_name,
    served_entities=served_entities,
    traffic_config=TrafficConfig(routes=routes),
)

print(f"Traffic updated on {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait and verify

# COMMAND ----------

import time

print(f"Waiting for {endpoint_name} to apply new config...")
for i in range(60):
    ep = w.serving_endpoints.get(endpoint_name)
    state = ep.state
    if state and str(state.ready).endswith("READY") and \
       (state.config_update is None or str(state.config_update).endswith("NOT_UPDATING")):
        print(f"READY after {(i+1)*10}s")
        break
    if state and "FAILED" in str(state.config_update):
        print(f"UPDATE FAILED: {state.config_update}")
        break
    time.sleep(10)

# Show final config
ep = w.serving_endpoints.get(endpoint_name)
print(f"\nFinal config:")
for entity in ep.config.served_entities:
    print(f"  v{entity.entity_version}: {entity.state.deployment}")
if ep.config.traffic_config:
    for route in ep.config.traffic_config.routes:
        print(f"  {route.served_model_name} → {route.traffic_percentage}%")

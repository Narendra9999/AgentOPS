# AgentOPS — Session History & Memory Setup Guide

## Overview

AgentOPS supports three session/memory configurations. Choose based on your deployment target and requirements:

| Config | File | UC Delta | Lakebase | Long-Term Memory | Best For |
|--------|------|:---:|:---:|:---:|----------|
| **UC Only** | `config-uc-only.yaml` | yes | no | no | Databricks Apps, air-gapped environments |
| **Lakebase (Full)** | `config-lakebase.yaml` | yes | yes | yes | Databricks Apps with stateful conversations |
| **In-Memory** | `config-inmemory.yaml` | no | no | no | Model Serving, simple Q&A, demos, zero infra |

### Two deployment paths for the agent

The same agent can be deployed two ways. The deployment target determines what session/memory features are available:

| | Model Serving Endpoint | Databricks App |
|--|--|--|
| **Deployed via** | `agents.deploy()` / pipeline step 6 | `databricks bundle deploy` + app resource |
| **Container identity** | `model-serving` system auth | Named service principal (fully grantable) |
| **UC Table READ** | Yes (via SQL warehouse + declared resources) | Yes (SP granted SELECT) |
| **UC Table WRITE** | No (system auth gets read-only table access) | Yes (SP granted MODIFY) |
| **Lakebase access** | System auth: limited. OBO: No. | Yes (first-class app resource) |
| **SQL Warehouse access** | Yes (declare as resource) | Yes (SP granted CAN_USE) |
| **LLM / Vector Search** | Yes | Yes |
| **Spark** | No | No |
| **Best config** | `config-inmemory.yaml` (stateless) | `config-lakebase.yaml` or `config-uc-only.yaml` |

**Key insight:** Model Serving system auth can **read** UC tables and call SQL warehouses when declared as resources (`DatabricksTable`, `DatabricksSQLWarehouse` in `mlflow.pyfunc.log_model`). However, **write (MODIFY)** access to UC tables is not granted by resource declarations alone — the system auth identity doesn't inherit `account users` grants. For stateful agents that write conversation history, **use the Databricks App**.

**Model Serving audit data** is captured automatically via:
- **Inference tables** (`inference_tables_enabled: true`) — every request/response logged
- **MLflow traces** — every request generates a trace in the experiment
- **Pipeline eval** — smoke test + post-deployment eval write to UC via Spark

### Where each backend works at serving time

| Backend | Notebook/Cluster | Model Serving | Databricks App |
|---------|:---:|:---:|:---:|
| UC Delta READ (SQL API) | yes | yes (declared resources) | yes |
| UC Delta WRITE (SQL API) | yes | **no** (system auth = read-only) | yes |
| UC Delta READ/WRITE (Spark) | yes | no (no Spark) | no (no Spark) |
| Lakebase (DatabricksStore) | yes | no (TCP timeout) | yes |
| Inference tables (auto-capture) | n/a | yes (automatic) | n/a |
| In-Memory (stateless) | yes | yes | yes |

**Model Serving limitation:** The `model-serving` system auth gets **read-only** access to UC tables via declared resources. Write (MODIFY) operations fail with `PERMISSION_DENIED`. This is a platform-level restriction — the system identity doesn't inherit `account users` grants.

### Two options for persistent history with Model Serving

If you need persistent session history with a Model Serving endpoint:

**Option A: Use a Databricks App instead (recommended)**

Deploy the agent as a Databricks App (`config-lakebase.yaml`). The app has a named service principal with full UC + Lakebase access. This gives you the richest experience: conversation continuity, long-term user memory, UC audit trail, and a chat UI.

**Option B: Client-managed history (middleware pattern)**

Keep the Model Serving endpoint **stateless** (`config-inmemory.yaml`) and have your calling application or middleware own the session history:

```
┌──────────────┐     ┌─────────────────────┐     ┌───────────────────┐
│  Frontend /  │────▶│  Middleware (App,    │────▶│  Model Serving    │
│  Chat UI     │     │  Lambda, or API)     │     │  Endpoint         │
└──────────────┘     └─────────────────────┘     │  (stateless)      │
                           │                      └───────────────────┘
                           │  Reads/writes
                           │  session history
                           ▼
                     ┌─────────────────────┐
                     │  UC Delta Table     │
                     │  (session_history)  │
                     └─────────────────────┘
```

How it works:
1. **Middleware** receives the user's message + `thread_id`
2. **Middleware** reads prior conversation from UC table (via SQL warehouse or Spark)
3. **Middleware** passes full message history to Model Serving in the request
4. **Model Serving** processes statelessly (uses `config-inmemory.yaml`)
5. **Middleware** saves user message + assistant response back to UC table
6. **Middleware** returns response to the frontend

This pattern keeps the serving endpoint simple and scalable while centralizing session state in the middleware layer. The middleware can be a Databricks App, a separate FastAPI service, or any application with UC access.

### Audit trail options

| Source | Model Serving | Databricks App | Clusters |
|--------|:---:|:---:|:---:|
| **UC Delta `session_history` table** | yes (via SQL API, if declared) | yes (via SQL API) | yes (via Spark) |
| **Inference tables** (automatic) | yes | n/a | n/a |
| **MLflow traces** | yes | yes | yes |
| **Pipeline eval** (Spark) | yes (runs on cluster) | n/a | yes |

### To switch configs

```bash
cd src/agent_development/agent/

# Option 1: UC only
cp config-uc-only.yaml config.yaml

# Option 2: Lakebase + UC + long-term memory
cp config-lakebase.yaml config.yaml

# Option 3: Stateless (no persistence)
cp config-inmemory.yaml config.yaml

# Then deploy
databricks bundle deploy -t dev
```

---

## Option 1: UC Only Setup

### What it does
- Appends every conversation turn to a Delta table (`{catalog}.{schema}.session_history`)
- Reads history back for multi-turn continuity
- On clusters: uses Spark SQL (fastest)
- On Databricks Apps: falls back to SQL Statement Execution API (no Spark needed)
- On Model Serving: **not available** (system auth can't access SQL warehouses — use `config-inmemory.yaml` instead)

### Prerequisites

1. **SQL Warehouse** (serverless recommended — instant startup, scale to zero)

   Create via UI: SQL Warehouses > Create > Serverless > X-Small
   
   Or note the ID of an existing one.

2. **UC Permissions** for the service principal (Model Serving / App SP):

   ```sql
   -- Grant on catalog
   GRANT USE_CATALOG ON CATALOG <catalog> TO `<service_principal_id>`;
   
   -- Grant on schema
   GRANT USE_SCHEMA ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
   GRANT CREATE_TABLE ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
   GRANT MODIFY ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
   GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
   ```

3. **SQL Warehouse permission** for the service principal:

   SQL Warehouses > (your warehouse) > Permissions > Add > `<service_principal>` > CAN USE

### Config (`config-uc-only.yaml`)

```yaml
session_history:
  enabled: true
  unity_catalog:
    enabled: true
    table: session_history                # Creates {catalog}.{schema}.session_history
    warehouse_id: 3c623898df7d3c96        # Your serverless warehouse ID (or "auto")
  lakebase:
    enabled: false
long_term_memory:
  enabled: false
```

### Table schema (auto-created)

| Column | Type | Description |
|--------|------|-------------|
| turn_id | STRING | Unique turn identifier (UUID) |
| session_id | STRING | Thread/session ID |
| turn_number | INT | Turn sequence number |
| user_message | STRING | User's question |
| assistant_response | STRING | Agent's response |
| request_time | STRING | ISO 8601 timestamp |
| response_time_ms | DOUBLE | Latency in milliseconds |
| model_endpoint | STRING | LLM endpoint used |
| trace_id | STRING | MLflow trace ID |
| metadata | STRING | Additional metadata (JSON string) |

### Query session history

```sql
-- Recent conversations
SELECT session_id, turn_number, user_message, LEFT(assistant_response, 100),
       request_time, response_time_ms
FROM <catalog>.<schema>.session_history
ORDER BY request_time DESC
LIMIT 20;

-- Specific thread
SELECT * FROM <catalog>.<schema>.session_history
WHERE session_id = '<thread_id>'
ORDER BY turn_number;

-- Avg response time by day
SELECT DATE(request_time) as day, AVG(response_time_ms) as avg_latency_ms, COUNT(*) as turns
FROM <catalog>.<schema>.session_history
GROUP BY 1 ORDER BY 1 DESC;
```

---

## Option 2: Lakebase (Full Memory) Setup

### What it does
- **Short-term**: Per-thread conversation state via Lakebase DatabricksStore (sub-ms reads)
- **Long-term**: Cross-session user facts with semantic search (remembers user's name, role, preferences)
- **Audit trail**: Also writes to UC Delta table for analytics
- **Agent tools**: LLM can call `save_memory` / `recall_memories` to store user facts

### Prerequisites

#### A. Create Lakebase Autoscaling Project

1. Go to **Compute > Lakebase** in the Databricks workspace
2. Click **Create Project**
3. Name: `agentops-sessions` (or your preferred name)
4. The `production` branch is created automatically
5. Note: No tables to create — `DatabricksStore.setup()` handles internal schema

#### B. Grant App Service Principal Lakebase Access

1. Go to **Compute > Lakebase > agentops-sessions > Roles**
2. Click **Add Role**
3. Select the app's service principal (e.g., `app-XXXX agentops-docs-dev`)
4. Role: `DATABRICKS_SUPERUSER` (needed for store.setup() to create internal tables)

#### C. UC Permissions (same as UC Only)

```sql
GRANT USE_CATALOG ON CATALOG <catalog> TO `<service_principal_id>`;
GRANT USE_SCHEMA ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
GRANT CREATE_TABLE ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
GRANT MODIFY ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<service_principal_id>`;
```

#### D. SQL Warehouse + Vector Search Endpoint Permissions

```
SQL Warehouses > (warehouse) > Permissions > Add > <app SP> > CAN USE
Vector Search > (endpoint) > Permissions > Add > <app SP> > CAN USE
```

### Config (`config-lakebase.yaml`)

```yaml
session_history:
  enabled: true
  unity_catalog:
    enabled: true
    table: session_history
    warehouse_id: 3c623898df7d3c96
  lakebase:
    enabled: true
    project: agentops-sessions           # Lakebase Autoscaling project name
    branch: production                   # Lakebase branch
long_term_memory:
  enabled: true
```

### Databricks App Deployment

The Lakebase config is best used with the **Databricks App** (not Model Serving, which can't reach Lakebase).

1. Uncomment `app-resource.yml` in `databricks.yml`:
   ```yaml
   include:
     - resources/app-resource.yml
   ```

2. Update `resources/app-resource.yml` with your Lakebase project/branch and serving endpoint names

3. Update `src/agent_deployment/app/app.yaml` environment variables:
   ```yaml
   env:
     - name: LAKEBASE_AUTOSCALING_PROJECT
       value: agentops-sessions
     - name: LAKEBASE_AUTOSCALING_BRANCH
       value: production
     - name: UC_SESSION_ENABLED
       value: "true"
     - name: SQL_WAREHOUSE_ID
       value: <your_warehouse_id>
   ```

4. Deploy:
   ```bash
   databricks bundle deploy -t dev
   databricks api post /api/2.0/apps/<app-name>/deployments --json '{
     "source_code_path": "/Workspace/Users/<user>/.bundle/agentops/dev/files/src/agent_deployment/app",
     "mode": "SNAPSHOT"
   }'
   ```

### Memory Features

| Feature | How it works |
|---------|-------------|
| **Session continuity** | Conversation state stored in Lakebase, keyed by `thread_id`. User can continue a thread across page reloads. |
| **Cross-thread recap** | Conversation summary saved to long-term memory after each turn. User can ask "recap last conversation" on a new thread. |
| **User facts** | LLM calls `save_memory` tool when user shares personal info (name, role, preferences). Facts persist across all sessions. |
| **Semantic recall** | Relevant user memories auto-loaded before each response via `store.search()`. LLM uses them silently to personalize answers. |

---

## Option 3: In-Memory (Stateless) Setup

**This is the recommended config for Model Serving endpoints.**

### What it does
- No server-side persistence at all
- Multi-turn handled by the client passing full message history in each request
- Each request is independent — no state between calls
- Audit trail comes from **inference tables** (automatic) and **MLflow traces**

### Why this is the right choice for Model Serving
Model Serving containers run with a restricted `model-serving` system identity that:
- Cannot access SQL Warehouses (so UC Delta writes fail)
- Cannot reach Lakebase PG endpoints (TCP timeout from serverless containers)
- CAN call LLM endpoints and vector search (these work fine)

You still get full observability via:
- **Inference tables**: auto-capture every request/response (`inference_tables_enabled: true` in databricks.yml)
- **MLflow traces**: every request generates a trace in the experiment
- **Pipeline eval**: smoke test + post-deployment eval write to UC `session_history` via Spark (runs on cluster)

### Prerequisites
- None. No Lakebase, no SQL warehouse, no Delta tables needed.

### Config (`config-inmemory.yaml`)

```yaml
session_history:
  enabled: false
  unity_catalog:
    enabled: false
  lakebase:
    enabled: false
long_term_memory:
  enabled: false
```

### How multi-turn works (client-managed)

The client sends the full conversation history in each request:

```json
{
  "messages": [
    {"role": "user", "content": "What is Delta Lake?"},
    {"role": "assistant", "content": "Delta Lake is..."},
    {"role": "user", "content": "How does time travel work?"}
  ]
}
```

The agent uses `max_history_turns` (config: `llm.max_history_turns`) to truncate if too long.

---

## Troubleshooting

### UC Delta: "No SQL warehouse available"
- Set `warehouse_id` explicitly in config (don't rely on `auto` from Model Serving)
- Or set the `SQL_WAREHOUSE_ID` environment variable

### UC Delta: Permission denied on INSERT
- Grant `MODIFY` and `CREATE_TABLE` on the schema to the service principal
- Grant `CAN_USE` on the SQL warehouse

### Lakebase: "couldn't get a connection after 30.00 sec"
- This happens on **Model Serving** — serverless containers can't reach Lakebase PG endpoints
- Use the **Databricks App** deployment instead, or switch to UC-only config

### Lakebase: "branch id not found"
- Verify the branch name matches exactly (default: `production`)
- Check the Lakebase project exists: Compute > Lakebase > project name

### Lakebase: "InvalidNamespaceError: Namespace labels cannot contain periods"
- User IDs with dots (emails like `user@company.com`) are auto-sanitized
- The framework replaces `.` with `_` and `@` with `_at_` in namespace labels

### PII guardrail blocking responses
- Emails from documentation context are whitelisted automatically
- Common domains (`@databricks.com`, `@example.com`) are in the safe list
- User's own email (from `user_id`) is whitelisted

### Intent guardrail blocking follow-ups on new threads
- Recalled long-term memories provide context for the intent check
- If user has prior memories with Databricks keywords, follow-ups like "recap" won't be blocked

---

## Architecture Reference

```
Request flow (Databricks App — Lakebase config):

  User → /invocations
    │
    ├─ 1. Load session history (Lakebase → short-term)
    ├─ 2. Recall user memories (Lakebase → long-term, semantic search)
    ├─ 3. Pre-LLM guardrails (with memory context for intent check)
    ├─ 4. Vector search (retrieve docs)
    ├─ 5. Build augmented prompt (system + docs + memories + history)
    ├─ 6. Call LLM (with memory tools if user_id provided)
    │     └─ Tool loop: save_memory / recall_memories
    ├─ 7. Post-LLM guardrails (PII whitelist, compliance, quality)
    ├─ 8. Save session history (Lakebase)
    ├─ 9. Save conversation summary (Lakebase → long-term)
    └─ 10. UC audit trail (SQL Statement API → Delta table)
    │
  Response ← assistant message + thread_id

Request flow (Model Serving — UC-only config):

  User → /invocations
    │
    ├─ 1. Load session history (UC Delta → SQL Statement API)
    ├─ 2. Pre-LLM guardrails
    ├─ 3. Vector search (retrieve docs)
    ├─ 4. Build augmented prompt (system + docs + history)
    ├─ 5. Call LLM
    ├─ 6. Post-LLM guardrails
    └─ 7. Save turn (UC Delta → SQL Statement API)
    │
  Response ← assistant message + thread_id
```

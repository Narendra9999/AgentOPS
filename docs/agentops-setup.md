# AgentOPS Setup Guide

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Clone and Configure](#step-1-clone-and-configure)
- [Step 2: Configure Authentication](#step-2-configure-authentication)
- [Step 3: Configure Unity Catalog](#step-3-configure-unity-catalog)
- [Step 4: Run Unit Tests](#step-4-run-unit-tests)
- [Step 5: Validate and Deploy](#step-5-validate-and-deploy)
- [Step 6: Run the Pipeline](#step-6-run-the-pipeline)
- [Step 7: Setup Monitoring](#step-7-setup-monitoring)
- [Step 8: Iterative Improvement](#step-8-iterative-improvement)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## Overview

AgentOPS is a framework for productionizing AI agents on Databricks. One pipeline does everything — data prep, model registration, endpoint creation, and evaluation — with every step tracked in audit tables.

### What Gets Deployed

```
databricks bundle deploy → creates:

  1 cluster     agentops-dev (ML runtime, UC-enabled)
  1 pipeline    agentops-pipeline (6 chained tasks — the main flow)
  3 cron jobs   monitoring (hourly), audit (daily), batch inference (daily)
  1 wheel       agentops-framework-0.1.0.whl
  10 notebooks  all code uploaded to workspace
```

### The Pipeline (one job, 6 chained tasks)

```
databricks bundle run agentops_pipeline -t dev

  Step 1: data_ingestion        ← Scrape Databricks docs from sitemap
      │
  Step 2: data_preprocessing    ← Clean HTML, chunk text (sentence/semantic/fixed)
      │
  Step 3: vector_search_setup   ← Create VS endpoint + delta sync index
      │
  Step 4: register_model        ← Register agent in Unity Catalog
      │
  Step 5: create_endpoint       ← Create serving endpoint (champion/challenger)
      │
  Step 6: evaluation            ← Quality gate: guardrails + MLflow eval

  If any step fails → downstream steps skip → audit logs FAILED
```

Every step is tracked in the `pipeline_step_log` audit table with timing, records processed, and status.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python 3.10+ | For local tests |
| Databricks CLI 0.230+ | `pip install databricks-cli` |
| Workspace access | Admin access to DEV (+ STAGE for promotion) |
| Unity Catalog | Catalog created in the workspace |

---

## Step 1: Clone and Configure

```bash
git clone <your-repo-url>
cd AgentOPS
```

### Edit `agentops_demo/databricks.yml`

Key variables to set:

```yaml
variables:
  catalog:
    default: <your-catalog>         # e.g., classic_stable_cykcbe_catalog
  schema:
    default: agentops
  llm_endpoint:
    default: gpt-oss-120b           # or your preferred LLM
  embedding_model:
    default: databricks-gte-large-en

targets:
  dev:
    workspace:
      host: <your-dev-workspace-url>
```

### Edit `agentops_demo/agent_development/agent/config.yaml`

Customize: system prompt, guardrail rules, intent keywords, LlamaGuard settings.

---

## Step 2: Configure Authentication

```bash
databricks auth login <your-workspace-url> --profile=dev
databricks auth profiles | grep dev    # verify: should show YES
```

### For STAGE (CI/CD)

Create service principals per environment:
1. **Account Console** → User Management → Service Principals
2. Create `agentops-dev-sp`, `agentops-stage-sp`
3. Generate OAuth secrets
4. Add SPs to respective workspaces

---

## Step 3: Configure Unity Catalog

```sql
CREATE SCHEMA IF NOT EXISTS <catalog>.agentops;
CREATE SCHEMA IF NOT EXISTS <catalog>.agentops_audit;

-- Grant to service principal
GRANT USE_CATALOG ON CATALOG <catalog> TO `<sp-name>`;
GRANT USE_SCHEMA, CREATE_TABLE, CREATE_MODEL, MODIFY, SELECT
  ON SCHEMA <catalog>.agentops TO `<sp-name>`;
GRANT USE_SCHEMA, CREATE_TABLE, MODIFY, SELECT
  ON SCHEMA <catalog>.agentops_audit TO `<sp-name>`;
```

---

## Step 4: Run Unit Tests

```bash
python -m pytest tests/unit/ -v
```

**51 tests** covering:
- Pre-LLM guardrails (18): length, PII, injection, toxicity, intent
- Post-LLM guardrails (11): toxicity, compliance, PII leakage, hallucination
- Chunking strategies (16): fixed, sentence, semantic, HTML cleaning

All tests run locally — no Databricks connection needed (mlflow/SDK are mocked).

---

## Step 5: Validate and Deploy

```bash
# Validate bundle
databricks bundle validate -t dev --profile=dev

# Deploy
databricks bundle deploy -t dev --profile=dev
```

### What gets created

| Resource | Name | Type |
|----------|------|------|
| Cluster | `agentops-dev` | ML runtime, UC, autoscale 1-3 |
| **Pipeline** | `agentops-pipeline-dev` | **6 chained tasks** (main flow) |
| Monitoring | `agentops-monitoring-dev` | Hourly cron |
| Audit aggregation | `agentops-audit-aggregation-dev` | Daily cron |
| Batch inference | `agentops-batch-inference-dev` | Daily 6am cron |

---

## Step 6: Run the Pipeline

```bash
databricks bundle run agentops_pipeline -t dev --profile=dev
```

This runs all 6 steps in order. Each step is tracked in the audit table:

| Step | Task | What It Does | Audit Entry |
|------|------|-------------|-------------|
| 1 | `data_ingestion` | Scrape docs sitemap → `databricks_docs_raw` table | records_processed, duration |
| 2 | `data_preprocessing` | Clean + chunk → `databricks_docs_chunked` table | chunk count, strategy used |
| 3 | `vector_search_setup` | Create VS endpoint + index | index name, endpoint status |
| 4 | `register_model` | Register agent in UC | model version, URI |
| 5 | `create_endpoint` | Create serving endpoint | endpoint name, workload size |
| 6 | `evaluation` | Guardrail + quality eval → per-row results | pass/fail, scores, evaluation_id |

### Monitor progress

Open the job run URL printed by the CLI, or:
```sql
-- Check audit trail
SELECT * FROM <catalog>.agentops_audit.pipeline_step_log
WHERE execution_id = '<latest>'
ORDER BY step_order;
```

### If a step fails

- Downstream steps are **automatically skipped**
- `pipeline_execution_log` records `status = FAILED`
- Fix the issue and re-run: `databricks bundle run agentops_pipeline -t dev`

---

## Step 7: Setup Monitoring

### Automatic (deployed with the bundle)

| Job | Schedule | What It Does |
|-----|----------|-------------|
| `agentops-monitoring-dev` | Every hour | Latency p50/p95, guardrail block rates, drift detection |
| `agentops-audit-aggregation-dev` | Daily midnight | Aggregates guardrail stats into summary |
| `agentops-batch-inference-dev` | Daily 6am | Processes `batch_input` table through endpoint |

### Optional: Lakehouse Monitoring dashboard

Open in workspace:
```
agent_deployment/monitoring/notebooks/SetupLakehouseMonitoring
```
Creates an auto-refreshing dashboard on the inference table (latency, volume, drift).

---

## Step 8: Iterative Improvement

After the agent is deployed and receiving real queries:

1. **Experts label traces** in MLflow UI (thumbs up/down + rationale)
2. **Run IterativeImprovement.py** notebook:
   - Collects expert feedback
   - Aligns LLM judge with experts (MemAlign)
   - Optimizes system prompt (GEPA)
3. **Review** the optimized prompt → update `config.yaml`
4. **Re-run pipeline** to redeploy with improved prompt

---

## Configuration Reference

### 3 Config Files

| File | Who Edits | What |
|------|-----------|------|
| `agentops_demo/databricks.yml` | Platform/DevOps | Workspaces, catalogs, compute, table names, scaling |
| `agent_development/agent/config.yaml` | Data scientist | System prompt, guardrails, intent keywords, LLM settings |
| `framework/guardrails/llamaguard_categories.yaml` | Security/Compliance | Safety categories to block/allow (S1-S14) |

### Key Variables in `databricks.yml`

**Core:** catalog, schema, audit_schema, agent_name, llm_endpoint (gpt-oss-120b), embedding_model

**Data Preparation:** data_source_url, max_documents, raw_data_table, preprocessed_data_table, chunk_size, chunk_overlap, min_chunk_size, chunking_strategy (fixed/sentence/semantic)

**Vector Search:** vs_index, vs_search_type (similarity/hybrid/mmr), vs_num_results, vs_reranker_enabled, vs_reranker_model

**Evaluation:** eval_golden_table, eval_adversarial_table, eval_results_table

**Serving:** chatbot_name, champion_model_version, champion_workload_size, champion_traffic_percentage, challenger_enabled, serving_scale_to_zero

**Inference/Monitoring:** inference_tables_enabled, rate_limit_per_user_per_minute, usage_tracking_enabled, batch_schedule

### Guardrails — Two Modes

**Keyword-based** (fast, free): `pii → injection → toxicity → intent`

**LlamaGuard** (accurate, context-aware): `pii → llamaguard → intent`

Set in `config.yaml`:
```yaml
guardrails:
  pre_llm:
    llamaguard_enabled: true       # false for keyword mode
    enabled_checks: [input_length_min, input_length_max, pii, llamaguard, intent]
```

### Audit Tables (5 tables)

| Table | What It Tracks |
|-------|---------------|
| `pipeline_execution_log` | One row per pipeline run (status, duration) |
| `pipeline_step_log` | One row per step (timing, records, output) |
| `deployment_events` | DEV→STAGE promotions |
| `guardrail_audit_log` | Every block/pass event |
| `eval_results` | Per-row evaluation scores (queryable) |

---

## Project Structure

```
AgentOPS/
├── databricks.yml                         # Root: bundle name + includes
├── agentops_demo/
│   ├── databricks.yml                     # Variables, targets, artifacts
│   ├── framework/                         # Standardized core (→ wheel)
│   │   ├── agent_base.py                  # Base agent (guardrails + tracing)
│   │   ├── guardrails/                    # pre_llm, post_llm, llamaguard, categories.yaml
│   │   ├── evaluation/                    # evaluation_pipeline + save_eval_results
│   │   ├── audit/                         # audit_logger (5 tables + PipelineStepLogger)
│   │   ├── monitoring/                    # trace_monitor (metrics + drift)
│   │   ├── optimization/                  # prompt_optimizer (MemAlign + GEPA)
│   │   └── inference/                     # batch_runner (ai_query + quarantine)
│   ├── data_preparation/                  # Ingestion → preprocessing → vector search
│   ├── agent_development/                 # Agent code + config + eval datasets + notebooks
│   ├── agent_deployment/                  # ModelServing + batch + Lakehouse monitoring
│   └── resources/
│       ├── cluster-resource.yml           # Compute
│       ├── pipeline-resource.yml          # Main pipeline (6 chained tasks)
│       ├── monitoring-resource.yml        # Hourly monitoring + daily audit
│       └── batch-inference-resource.yml   # Daily batch inference
├── tests/
│   ├── unit/ (51 tests)
│   ├── smoke/ (2 tests)
│   └── integration/ (9 tests)
└── docs/
```

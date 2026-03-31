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
- [Air-Gapped Deployment (Mastercard)](#air-gapped-deployment-mastercard)
- [Project Structure](#project-structure)

---

## Overview

AgentOPS is a production framework for deploying AI agents on Databricks. One pipeline does everything — data prep, model registration, evaluation, endpoint deployment, smoke tests, and post-deployment validation — with every step tracked in audit tables.

**Framework Version:** 2.0.1

### What Gets Deployed

```
databricks bundle deploy → creates:

  1 pipeline    agentops-pipeline (8 chained tasks — the main flow)
  3 cron jobs   monitoring (hourly), audit (daily), batch inference (daily)
  1 wheel       agentops_framework-2.0.1-*.whl
  12 notebooks  all code uploaded to workspace
```

### The Pipeline (one job, 8 chained tasks)

```
databricks bundle run agentops_pipeline -t dev

  Step 1: data_ingestion          ← Scrape/load docs → raw table
      │
  Step 2: data_preprocessing      ← Clean HTML, chunk text → chunked table
      │
  Step 3: vector_search_setup     ← Create VS endpoint + delta sync index
      │
  Step 4: register_model          ← Build wheel, log model, register in UC, set @champion
      │
  Step 5: pre_deployment_eval     ← Load via pyfunc, LLM-as-judge eval, quality gate
      │
  Step 6: deploy_agent            ← Deploy endpoint (champion/challenger traffic splits)
      │
  Step 7: smoke_test              ← Validate live endpoint (8 tests)
      │
  Step 8: post_deployment_eval    ← Evaluate live endpoint, LLM-as-judge scores

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

### Edit `databricks.yml` (project root)

Key variables to set:

```yaml
variables:
  catalog:
    default: <your-catalog>         # e.g., classic_stable_cykcbe_catalog
  schema:
    default: agentops
  llm_endpoint:
    default: databricks-gpt-oss-120b  # or your preferred LLM
  embedding_model:
    default: databricks-gte-large-en

targets:
  dev:
    workspace:
      host: <your-dev-workspace-url>
    variables:
      cluster_id: "<your-cluster-id>"
```

### Edit `src/agent_development/agent/config.yaml`

Customize: system prompt, guardrail rules, intent keywords, domain keywords.

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

Tests cover:
- Pre-LLM guardrails: length, PII, injection, toxicity, intent
- Post-LLM guardrails: toxicity, compliance, PII leakage, hallucination, quality
- Chunking strategies: fixed, sentence, semantic, HTML cleaning

All tests run locally — no Databricks connection needed (mlflow/SDK are mocked).

---

## Step 5: Validate and Deploy

```bash
# Validate bundle
databricks bundle validate -t dev

# Deploy
databricks bundle deploy -t dev
```

### What gets created

| Resource | Name | Type |
|----------|------|------|
| **Pipeline** | `agentops-pipeline-dev` | **8 chained tasks** (main flow) |
| Monitoring | `agentops-monitoring-dev` | Hourly cron |
| Audit aggregation | `agentops-audit-aggregation-dev` | Daily cron |
| Batch inference | `agentops-batch-inference-dev` | Daily 6am cron |

All tasks run on an existing cluster specified by `cluster_id` in `databricks.yml`.

---

## Step 6: Run the Pipeline

```bash
databricks bundle run agentops_pipeline -t dev
```

This runs all 8 steps in order. Each step is tracked in the audit table:

| Step | Task | What It Does | Audit Entry |
|------|------|-------------|-------------|
| 1 | `data_ingestion` | Scrape/load docs → `databricks_docs_raw` table | records_processed, duration |
| 2 | `data_preprocessing` | Clean + chunk → `databricks_docs_chunked` table | chunk count, strategy used |
| 3 | `vector_search_setup` | Create VS endpoint + delta sync index | index name, endpoint status |
| 4 | `register_model` | Build wheel, log model, register in UC, set @champion | model version, URI |
| 5 | `pre_deployment_eval` | Load model via pyfunc, LLM-as-judge eval, quality gate | pass/fail, scores |
| 6 | `deploy_agent` | Deploy serving endpoint (champion/challenger) | endpoint name, traffic config |
| 7 | `smoke_test` | Validate live endpoint (8 tests) | test results, pass/fail |
| 8 | `post_deployment_eval` | Evaluate live endpoint, LLM-as-judge scores | evaluation_id, scores |

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
src/agent_deployment/monitoring/notebooks/SetupLakehouseMonitoring
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

### 2 Config Files

| File | Who Edits | What |
|------|-----------|------|
| `databricks.yml` | Platform/DevOps | Workspaces, catalogs, compute, table names, scaling, AI Gateway |
| `src/agent_development/agent/config.yaml` | Data scientist | System prompt, guardrails, intent keywords, LLM settings |

### Config Architecture

```
databricks.yml  → Infrastructure: workspaces, catalogs, endpoints, cluster_id
config.yaml     → Agent behavior: LLM, prompt, guardrails, history turns
RegisterModel.py merges both → runtime_config.yaml baked into model artifact
```

`databricks.yml` is the single source of truth for infra settings. `config.yaml` values are overridden by `databricks.yml` at deploy time.

### Key Variables in `databricks.yml`

**Core:** catalog, schema, audit_schema, agent_name, cluster_id, llm_endpoint, embedding_model

**Data Preparation:** data_source_url, max_documents, raw_data_table, preprocessed_data_table, chunk_size, chunk_overlap, min_chunk_size, chunking_strategy (fixed/sentence/semantic)

**Vector Search:** vs_endpoint, vs_index, vs_search_type (similarity/hybrid/mmr), vs_num_results, vs_reranker_enabled, vs_reranker_model

**Evaluation:** eval_golden_table, eval_adversarial_table, eval_results_table

**Serving:** chatbot_name, champion_model_version, champion_workload_size, champion_traffic_percentage, challenger_enabled, serving_scale_to_zero

**AI Gateway:** ai_gateway_safety_enabled, inference_tables_enabled, rate_limit_per_user_per_minute, rate_limit_per_endpoint_per_minute, usage_tracking_enabled

**Batch Inference:** inference_mode (online/batch/both), batch_input_table, batch_output_table, batch_quarantine_table, batch_schedule

**Monitoring:** lakehouse_monitoring_enabled, monitoring_granularity

### Guardrails

Pre-LLM checks (keyword-based, fast): `input_length → pii → injection → toxicity → intent`

Post-LLM checks: `toxicity → compliance → pii_leakage → hallucination → quality`

LLM-based content safety is handled by **AI Gateway safety filter** on the serving endpoint (`ai_gateway_safety_enabled: true` in `databricks.yml`), not by in-code LlamaGuard.

Set in `config.yaml`:
```yaml
guardrails:
  enabled: true
  pre_llm:
    enabled_checks:
      - input_length_min
      - input_length_max
      - pii
      - injection
      - toxicity
      - intent
```

### Evaluation Architecture (mlflow.genai.evaluate)

7 LLM-as-judge scorers via `@scorer` functions:

| Scorer | Scale | What It Measures |
|--------|-------|-----------------|
| accuracy | 1-5 | Factual correctness |
| helpfulness | 1-5 | Actionable, practical guidance |
| professionalism | 1-5 | Formal tone |
| docs_relevance | 1-5 | Databricks-specific content |
| code_snippet_quality | 1-5 | Code examples for coding questions |
| source_citation | 1-5 | References to documentation |
| answer_completeness | 1-5 | Thorough, non-deflecting answers |

Judge model: `databricks-meta-llama-3-3-70b-instruct` (internal FMAPI, no internet)

### Audit Tables (5 tables)

| Table | What It Tracks |
|-------|---------------|
| `pipeline_execution_log` | One row per pipeline run (status, duration) |
| `pipeline_step_log` | One row per step (timing, records, output) |
| `deployment_events` | DEV→STAGE promotions |
| `guardrail_audit_log` | Every block/pass event |
| `eval_results` | Per-row evaluation scores (queryable) |

---

## Deployment Targets

| Target | Workspace | Mode | Notes |
|--------|-----------|------|-------|
| `dev` | FEVM (classic_stable_cykcbe) | development | Default target, existing cluster |
| `e2-demo` | e2-demo-field-eng | development | Demo workspace |
| `mastercard` | Customer workspace | development | Air-gapped, bundled dataset |
| `stage` | Production staging | production | Service principal, CI/CD |

---

## Air-Gapped Deployment (Mastercard)

Mastercard's environment has no internet access. The framework handles this:

### Pip Install
- Conditional: auto-detects Mastercard volume → falls back to PyPI
- Volume: `/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/`
- `--no-index` flag prevents all PyPI calls
- `--no-build-isolation` for wheel build (uses cluster's setuptools)
- All notebooks with pip install also have `restartPython()` + widget re-read

### Data
- Bundled dataset: 4568 docs in `fixtures/databricks_docs.json` (14 MB)
- `data_source_url: "local"` in Mastercard target reads from bundled JSON
- No internet needed for data ingestion

### Evaluation
- `mlflow.genai.evaluate()` — no default evaluator, no tiktoken/HuggingFace downloads
- `@scorer` functions call internal Databricks FMAPI (no internet)
- Zero PyPI calls in production

### Vector Search
- Uses Databricks SDK only (`w.vector_search_indexes.query_index`)
- No `databricks-vectorsearch` package needed
- `get_or_create_endpoint` waits for ONLINE (600s)
- `create_delta_sync_index` waits for sync complete (1200s)

### Wheel Build
- Framework wheel embedded in model artifact (conda_env + artifacts pattern)
- UC Volume stores wheel for versioning/audit
- No external dependencies in pyproject.toml (all pre-installed on ML runtime)

### MLflow Compatibility
- MLflow 3.3.2 (Mastercard): metrics use `{name}/mean`, scores in `assessments[].feedback.value`
- MLflow 3.10+ (FEVM): metrics use `{name}/v1/mean`, scores in `{name}/value` columns
- Code handles both formats automatically

---

## Project Structure

```
AgentOPS/
├── databricks.yml                              # Bundle config: variables, targets, artifacts
├── pyproject.toml                              # Framework wheel (v2.0.1, no external deps)
├── requirements.txt
├── setup.py
│
├── src/
│   ├── framework/                              # Standardized core (→ wheel artifact)
│   │   ├── agent_base.py                       # AgentOPSBase(ChatAgent)
│   │   ├── mlops_utils.py                      # Mastercard MLOps utilities
│   │   ├── guardrails/
│   │   │   ├── pre_llm.py                      # 6 pre-LLM checks
│   │   │   └── post_llm.py                     # 5 post-LLM checks
│   │   ├── evaluation/
│   │   │   └── evaluation_pipeline.py          # run_evaluation() via mlflow.genai.evaluate()
│   │   ├── audit/
│   │   │   └── audit_logger.py                 # PipelineStepLogger + _safe_json_dumps
│   │   ├── monitoring/
│   │   │   └── trace_monitor.py                # Metrics, guardrail stats, drift
│   │   ├── optimization/
│   │   │   └── prompt_optimizer.py             # MemAlign + GEPA
│   │   └── inference/
│   │       └── batch_runner.py                 # ai_query() with quarantine
│   │
│   ├── data_preparation/
│   │   ├── data_ingestion/
│   │   │   ├── ingestion/
│   │   │   │   └── fetch_data.py               # fetch_data_from_url + load_data_from_file
│   │   │   └── notebooks/
│   │   │       └── DataIngestion.py
│   │   ├── data_preprocessing/
│   │   │   ├── preprocessing/
│   │   │   │   └── create_chunk.py             # Chunking strategies
│   │   │   └── notebooks/
│   │   │       └── DataPreprocessing.py        # → chunk_text, url, chunk_id
│   │   └── vector_search/
│   │       ├── vector_search_utils/
│   │       │   └── utils.py                    # SDK only, waits for ONLINE
│   │       └── notebooks/
│   │           └── VectorSearch.py
│   │
│   ├── agent_development/
│   │   ├── agent/
│   │   │   ├── config.yaml                     # Agent config (merged at deploy)
│   │   │   ├── notebooks/
│   │   │   │   ├── Agent.py                    # DatabricksDocsAgent (RAG)
│   │   │   │   ├── RegisterModel.py            # Log + register (no deploy)
│   │   │   │   └── DeployAgent.py              # Deploy endpoint (champion/challenger)
│   │   │   └── tools/
│   │   │       └── agent_tools.py              # search_docs (SDK)
│   │   └── agent_evaluation/
│   │       ├── evaluation/
│   │       │   ├── custom_scorers.py           # 7 @scorer LLM-as-judge functions
│   │       │   ├── golden_dataset.json
│   │       │   └── adversarial_dataset.json
│   │       └── notebooks/
│   │           ├── PreDeploymentEval.py         # pyfunc.load_model + genai.evaluate
│   │           ├── PostDeploymentEval.py        # Live endpoint + genai.evaluate
│   │           ├── SmokeTest.py                 # 8 endpoint tests
│   │           ├── RunMonitoring.py             # Hourly + Lakehouse Monitor
│   │           ├── IterativeImprovement.py      # MemAlign + GEPA
│   │           └── AggregateAudit.py            # Daily guardrail summary
│   │
│   └── agent_deployment/
│       ├── model_serving/
│       │   └── notebooks/
│       │       └── UpdateTraffic.py
│       ├── batch_inference/
│       │   └── notebooks/
│       │       └── RunBatchInference.py
│       └── monitoring/
│           └── notebooks/
│               └── SetupLakehouseMonitoring.py
│
├── resources/
│   ├── pipeline-resource.yml                   # 8-step pipeline
│   ├── monitoring-resource.yml                 # Hourly monitoring + daily audit
│   ├── batch-inference-resource.yml            # Daily batch job
│   └── serving-resource.yml.reference          # AI Gateway config (reference only)
│
├── fixtures/
│   ├── databricks_docs.json                    # 4568 docs bundled (14 MB)
│   ├── golden_dataset.json
│   └── adversarial_dataset.json
│
├── tests/
│   ├── unit/                                   # Pre/post guardrails + chunking
│   ├── smoke/                                  # Endpoint smoke tests
│   └── integration/                            # Serving endpoint tests
│
└── docs/
    ├── agentops-setup.md                       # This file
    └── examples/
        └── Serving Endpoint Query Examples.ipynb
```

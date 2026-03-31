---
name: AgentOPS Demo Scope
description: Mastercard AgentOPS framework — scope, status, architecture, and remaining work
type: skill
---

# AgentOPS — Mastercard Demo Scope

## Status Summary

| Area | Status |
|------|--------|
| Framework Version | **2.0.1** |
| 8-Step Pipeline | **All 8 steps passing on FEVM** |
| Monitoring Jobs | **Built** (hourly monitoring, daily audit) |
| Batch Inference Job | **Built** (daily cron) |
| Mastercard Target | **Config ready**, deployment in progress |
| CI/CD (Jenkinsfile) | **Not started** |
| Lakehouse Monitoring | **Notebook built**, dashboard setup pending |
| Documentation | **Updated** (agentops-setup.md, README.md) |

**Last validated:** 2026-03-30 (FEVM dev target, all 8 pipeline steps green)

---

## Architecture

```
databricks.yml (root)
  └── include: resources/*.yml
       ├── pipeline-resource.yml       → agentops-pipeline (8 chained tasks)
       ├── monitoring-resource.yml     → monitoring (hourly) + audit (daily)
       └── batch-inference-resource.yml → batch inference (daily 6am)

pyproject.toml → agentops_framework-2.0.1 wheel (no external deps)
```

### 8-Step Pipeline

```
Step 1: data_ingestion         → Scrape/load docs → raw table
Step 2: data_preprocessing     → Clean HTML, chunk → chunked table
Step 3: vector_search_setup    → VS endpoint + delta sync index
Step 4: register_model         → Build wheel, log model, register UC, @champion
Step 5: pre_deployment_eval    → pyfunc load, LLM-as-judge, quality gate
Step 6: deploy_agent           → Endpoint (champion/challenger traffic splits)
Step 7: smoke_test             → 8 endpoint validation tests
Step 8: post_deployment_eval   → Live endpoint LLM-as-judge scoring
```

### Cron Jobs (3)

| Job | Schedule | Notebook |
|-----|----------|----------|
| `agentops-monitoring` | Hourly | RunMonitoring.py |
| `agentops-audit-aggregation` | Daily midnight | AggregateAudit.py |
| `agentops-batch-inference` | Daily 6am | RunBatchInference.py |

---

## Project Structure (src/ layout)

```
AgentOPS/
├── databricks.yml                    # Bundle config (targets: dev, e2-demo, mastercard, stage)
├── pyproject.toml                    # Wheel v2.0.1 (no external deps)
├── src/
│   ├── framework/                    # Core → wheel artifact
│   │   ├── agent_base.py            # AgentOPSBase(ChatAgent)
│   │   ├── mlops_utils.py
│   │   ├── guardrails/{pre,post}_llm.py
│   │   ├── evaluation/evaluation_pipeline.py
│   │   ├── audit/audit_logger.py
│   │   ├── monitoring/trace_monitor.py
│   │   ├── optimization/prompt_optimizer.py
│   │   └── inference/batch_runner.py
│   ├── data_preparation/
│   │   ├── data_ingestion/           # fetch_data.py + DataIngestion notebook
│   │   ├── data_preprocessing/       # create_chunk.py + DataPreprocessing notebook
│   │   └── vector_search/            # utils.py (SDK only) + VectorSearch notebook
│   ├── agent_development/
│   │   ├── agent/                    # config.yaml, Agent.py, RegisterModel.py, DeployAgent.py
│   │   └── agent_evaluation/         # custom_scorers.py, PreDeploymentEval, PostDeploymentEval,
│   │                                 # SmokeTest, RunMonitoring, AggregateAudit, IterativeImprovement
│   └── agent_deployment/
│       ├── model_serving/            # UpdateTraffic.py
│       ├── batch_inference/          # RunBatchInference.py
│       └── monitoring/               # SetupLakehouseMonitoring.py
├── resources/                        # DAB resource YAMLs
├── fixtures/                         # Bundled data (4568 docs, golden/adversarial datasets)
├── tests/                            # unit, smoke, integration
└── docs/                             # Setup guide + examples
```

---

## Deployment Targets

| Target | Workspace | Catalog | Notes |
|--------|-----------|---------|-------|
| `dev` | FEVM (classic_stable_cykcbe) | classic_stable_cykcbe_catalog | Default, all 8 steps passing |
| `e2-demo` | e2-demo-field-eng | nmerla_agentops | Demo workspace |
| `mastercard` | Customer workspace | mc_edacde | Air-gapped, bundled dataset, TODOs for workspace URL/cluster/endpoints |
| `stage` | Production staging | env var | Service principal, CI/CD |

---

## Mastercard-Specific Constraints

- **Air-gapped**: No PyPI, no internet in production
- **Pip install**: Conditional path — detects Mastercard volume, falls back to `--no-index`
  - Volume: `/Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/`
  - `--no-build-isolation` for wheel build
  - `restartPython()` + widget re-read after pip installs
- **Data**: Bundled `fixtures/databricks_docs.json` (4568 docs, 14 MB), `data_source_url: "local"`
- **Evaluation**: `@scorer` functions call internal FMAPI, zero PyPI calls
- **Vector search**: SDK only (`w.vector_search_indexes.query_index`), no `databricks-vectorsearch` package
- **MLflow compat**: Handles both 3.3.2 (Mastercard) and 3.10+ (FEVM) metric formats automatically
- **Safety**: AI Gateway safety filter on endpoint (not in-code LlamaGuard)

---

## Evaluation Architecture

7 LLM-as-judge scorers via `mlflow.genai.evaluate()` + `@scorer`:

accuracy (1-5), helpfulness (1-5), professionalism (1-5), docs_relevance (1-5),
code_snippet_quality (1-5), source_citation (1-5), answer_completeness (1-5)

- Judge model: `databricks-meta-llama-3-3-70b-instruct` (internal FMAPI)
- Results saved to `eval_results` UC table (queryable)
- Quality gate: all scores must be 5/5 to pass pre-deployment

---

## Guardrails

Pre-LLM (keyword-based, fast): `input_length → pii → injection → toxicity → intent`
Post-LLM: `toxicity → compliance → pii_leakage → hallucination → quality`
Endpoint-level: AI Gateway safety filter (`ai_gateway_safety_enabled: true`)

---

## What's Been Built (completed)

- [x] Full 8-step pipeline (pipeline-resource.yml)
- [x] Framework wheel with guardrails, evaluation, audit, monitoring, optimization, batch inference
- [x] Agent (DatabricksDocsAgent) with RAG tool (SDK-based vector search)
- [x] Pre-deployment eval (pyfunc load, LLM-as-judge, quality gate)
- [x] Post-deployment eval (live endpoint, LLM-as-judge)
- [x] Smoke test (8 endpoint validation tests)
- [x] Model registration with @champion alias
- [x] Champion/challenger endpoint deployment with traffic splits
- [x] AI Gateway config (safety filter, rate limits, inference tables, usage tracking)
- [x] Monitoring job (hourly) — latency, guardrail stats, drift
- [x] Audit aggregation job (daily)
- [x] Batch inference job (daily) with quarantine for blocked records
- [x] Iterative improvement notebook (MemAlign + GEPA)
- [x] Air-gapped pip install handling
- [x] Bundled dataset for offline data ingestion
- [x] MLflow 3.3.2/3.10+ compatibility
- [x] Unit tests (pre/post guardrails, chunking)
- [x] SetupLakehouseMonitoring notebook
- [x] UpdateTraffic notebook
- [x] Setup documentation (agentops-setup.md)

## Remaining Work

- [ ] **Mastercard deployment**: Fill in TODOs (workspace URL, cluster ID, LLM/embedding endpoints) and run pipeline on Mastercard target
- [ ] **Lakehouse monitoring dashboard**: Run SetupLakehouseMonitoring notebook to create auto-refreshing dashboard
- [ ] **CI/CD Jenkinsfile**: Create Jenkinsfile for Mastercard's Jenkins-based CI/CD (validate → test → deploy → run pipeline)
- [ ] **README update**: Align README.md with current 8-step pipeline (still says 6 steps)
- [ ] **Unit test import fix**: Tests fail on collection (import errors) — need to fix conftest.py or mock dependencies

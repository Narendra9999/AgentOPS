---
name: AgentOPS Demo Scope
description: Mastercard AgentOPS framework — scope, status, architecture, and remaining work
type: skill
---

# AgentOPS — Mastercard Demo Scope

## Status Summary

| Area | Status |
|------|--------|
| Framework Version | **2.0.2** |
| 8-Step Pipeline (agentops) | **All 8 steps passing on FEVM** |
| 8-Step Pipeline (RAG) | **All 8 steps passing on FEVM** (databricks_rag schema) |
| Config-Driven Evaluation | **Complete** — builtin/llm_judge/domain/all modes, per-scorer weight/threshold/enabled |
| Session History | **Complete** — dual-backend (UC Delta + Lakebase DatabricksStore), config-driven |
| Long-Term Memory | **Complete** — cross-session user facts via Lakebase, agent-driven save/recall tools |
| Databricks App | **Complete** — FastAPI app with chat UI, Lakebase memory, vector search, MLflow traces |
| Guardrails | **Complete** — pre/post LLM, PII whitelist (docs + safe domains), intent w/ memory context |
| Monitoring Jobs | **Built** (hourly monitoring, daily audit) |
| Batch Inference Job | **Built** (daily cron) |
| Mastercard Target | **Config ready**, CI/CD pipeline running |
| CI/CD (Jenkins) | **CI passing** (unit tests), CD deploying |
| Lakehouse Monitoring | **Notebook built**, validation pending |
| Iterative Improvement | **Validated** — GEPA alignment running on FEVM (MLflow 3.10.1) |
| Documentation | **Updated** (agentops-setup.md, README.md) |

**Last validated:** 2026-04-08 (FEVM dev target, both pipelines green, Databricks App live)

---

## Architecture

```
databricks.yml (root)
  └── include: resources/*.yml
       ├── pipeline-resource.yml       → agentops-pipeline (8 chained tasks)
       ├── monitoring-resource.yml     → monitoring (hourly) + audit (daily)
       └── batch-inference-resource.yml → batch inference (daily 6am)

pyproject.toml → agentops_framework-2.0.2 wheel (no external deps)

resources/
  ├── pipeline-resource.yml         → agentops-pipeline (8 chained tasks, agentops schema)
  ├── rag-pipeline-resource.yml     → databricks-RAG-pipeline (8 tasks, databricks_rag schema)
  ├── monitoring-resource.yml       → monitoring (hourly) + audit (daily)
  ├── batch-inference-resource.yml  → batch inference (daily 6am)
  └── app-resource.yml              → Databricks App (FastAPI + Lakebase memory, opt-in)
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
│   │   ├── agent_base.py            # AgentOPSBase(ChatAgent) + session history hook
│   │   ├── mlops_utils.py
│   │   ├── guardrails/{pre,post}_llm.py
│   │   ├── evaluation/evaluation_pipeline.py
│   │   ├── session/session_store.py  # UC Delta + Lakebase DatabricksStore dual-backend
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
│   │   └── agent_evaluation/         # custom_scorers.py, scorer_loader.py, PreDeploymentEval,
│   │                                 # PostDeploymentEval, SmokeTest, RunMonitoring,
│   │                                 # AggregateAudit, IterativeImprovement
│   └── agent_deployment/
│       ├── app/                      # Databricks App (FastAPI + Lakebase + chat UI)
│       │   ├── server.py            # FastAPI server (/, /health, /invocations)
│       │   ├── agent_app.py         # Async agent with Lakebase memory + tools
│       │   ├── chat.html            # Browser chat UI
│       │   ├── app.yaml             # App config (env vars)
│       │   ├── requirements.txt     # App dependencies
│       │   ├── config.yaml          # Agent config (guardrails, system prompt)
│       │   └── guardrails/          # Pre/post LLM guardrails (app copy)
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

Config-driven via `config.yaml` → `evaluation.scorer_mode`:

| Mode | Scorers | LLM Calls |
|------|---------|-----------|
| `builtin` (default) | MLflow built-in: Guidelines, Correctness, RelevanceToQuery, Safety, RetrievalGroundedness | Yes (judge model) |
| `llm_judge` | 7 custom scorers: accuracy, helpfulness, professionalism, docs_relevance, code_snippet_quality, source_citation, answer_completeness (each with detailed 1-5 rubrics) | Yes (judge model) |
| `domain` | User-defined scorers: guidelines (LLM pass/fail), rubric (LLM 1-5), keyword (regex, no LLM), expected_facts (fact coverage, no LLM) | Depends on type |
| `all` | All three combined | Yes |

**Per-scorer customization**: `enabled`, `weight`, `threshold` (overrides global `quality_gate_threshold`)

- Judge model: `databricks-meta-llama-3-3-70b-instruct` (internal FMAPI)
- Scorer loader: `scorer_loader.py` → `load_scorers()`, `get_thresholds()`, `get_weights()`
- Results saved to `eval_results` UC table (queryable)
- Quality gate: configurable threshold (default 3.5), per-scorer overrides supported

---

## Guardrails

Pre-LLM (keyword-based, fast): `input_length → pii → injection → toxicity → intent`
Post-LLM: `toxicity → compliance → pii_leakage → hallucination → quality`
Endpoint-level: AI Gateway safety filter (`ai_gateway_safety_enabled: true`)

---

## What's Been Built (completed)

- [x] Full 8-step pipeline — agentops-pipeline (pipeline-resource.yml)
- [x] Full 8-step pipeline — databricks-RAG-pipeline (rag-pipeline-resource.yml, databricks_rag schema)
- [x] Framework wheel with guardrails, evaluation, audit, monitoring, optimization, batch inference, session history
- [x] Agent (DatabricksDocsAgent) with RAG tool (SDK-based vector search)
- [x] Config-driven evaluation scorers — builtin/llm_judge/domain/all modes
- [x] 4 domain scorer types: guidelines, rubric, keyword, expected_facts
- [x] Per-scorer customization: enabled, weight, threshold override
- [x] MLflow built-in scorers: Guidelines, Correctness, RelevanceToQuery, Safety, RetrievalGroundedness
- [x] 7 LLM-as-judge scorers with detailed 1-5 rubrics
- [x] Session history persistence — dual-backend (UC Delta table + Lakebase DatabricksStore)
- [x] Long-term agent memory — cross-session user facts via Lakebase DatabricksStore with semantic search
- [x] Databricks App — FastAPI server with chat UI, Lakebase memory, vector search, MLflow traces
- [x] Agent-driven memory tools — save_memory/recall_memories via LLM tool-calling loop
- [x] Cross-thread conversation summary — recap available on new threads via long-term memory
- [x] PII guardrail hardening — safe domain whitelist + doc-sourced email whitelist
- [x] Intent guardrail context — uses recalled memories as fallback for new-thread intent check
- [x] Pre-deployment eval (pyfunc load, config-driven scorers, quality gate)
- [x] Post-deployment eval (live endpoint, config-driven scorers)
- [x] Smoke test (8 endpoint validation tests, AI Gateway safety filter resilient)
- [x] Model registration with @champion alias
- [x] Champion/challenger endpoint deployment with traffic splits
- [x] AI Gateway config (safety filter, rate limits, inference tables, usage tracking)
- [x] Monitoring job (hourly) — latency, guardrail stats, drift
- [x] Audit aggregation job (daily)
- [x] Batch inference job (daily) with quarantine for blocked records
- [x] Iterative improvement notebook (GEPA alignment + prompt optimization)
- [x] Air-gapped pip install handling
- [x] Bundled dataset for offline data ingestion
- [x] MLflow 3.3.2/3.10+ compatibility
- [x] Unit tests (pre/post guardrails, chunking) — CI-only via testpaths config
- [x] SetupLakehouseMonitoring notebook
- [x] UpdateTraffic notebook
- [x] Setup documentation (agentops-setup.md)
- [x] Jenkins CI passing (unit tests only, integration/smoke excluded by default)

## Remaining Work (Next Steps)

- [ ] **Validate Lakehouse Monitoring job**: Run SetupLakehouseMonitoring + verify auto-refreshing dashboard
- [x] **Validate Iterative Improvement pipeline**: GEPA alignment validated on FEVM (2026-04-07) — 15 traces, expert labels, make_judge, GEPA alignment all working
- [ ] **Mastercard deployment**: Fill in TODOs (workspace URL, cluster ID, LLM/embedding endpoints) and run pipeline on Mastercard target
- [ ] **README update**: Align README.md with current architecture

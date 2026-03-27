Mastercard AgentOPS Framework Scope

Mastercard team is planning to implement a platform wide framework for productionizing agents from lower to production environment on databricks platform.
We are planning to implement a multi-phase approach to implement the framework before onboarding customers on to the platform.

Scope:
As part of the phase-1, we are planning to implement the below functionality as part of the framework.

Steps involved in the framework:
a) Data Preparation pipeline
b) Agent Development include the guardrails including prompt intent and toxicity before submitting to the llm and post response
c) Agent Evaluation should be guardrails.
d) Agent Deployment
e) Inference (should be able to support both Batch and Online for the user configuration)
f) Monitoring and Drift (Predeployment monitoring with dashboards)
g) Iterative Development with LLM as judge and prompt optimization ( use this blog as reference : https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks)

Code for the blog: https://github.com/WesleyPasfield/at-bat-assistant/tree/main

Mastercard uses Jenkins for their CI/CD process with bitbucket as their code repo.
Mastercard have multi-account setup for databricks platform. So, Each Environment -DEV,STAGE and PROD have been on their own accounts and are not workspaces on the same account.

For the initial phase-1 , we will just do DEV to STAGE deployment using DAB (Databricks Asset Bundles)

Out of scope:
a) MCP Servers hosting
b) Model serving using Databricks APP.
c) Other advanced patterns like multi-agent deployments ,etc.
d) Session Management using lakebase (designed, deferred)
e) Tool routing and LangGraph state machine

Usecase:
Read all the documentation of databricks and respond to user queries about the databricks products, integration, coding patterns and best practices and any code snippets/functions or capabilities defined databricks api documentation
data source : https://docs.databricks.com/en/doc-sitemap.xml
Reference: https://github.com/ryuta-yoshimatsu/agentops-demo


Implementation Status: ALL 8 STEPS PASSING ON FEVM + MASTERCARD DEPLOYMENT IN PROGRESS
=========================================================================================

GitHub Repo: https://github.com/narendra-merla_data/agentops-framework
Framework Version: 1.6.0


8-Step Pipeline Flow:
─────────────────────
  Step 1: data_ingestion         → Scrape/load docs → raw table
  Step 2: data_preprocessing     → Chunk documents → chunked table (chunk_text, url, chunk_id)
  Step 3: vector_search_setup    → Create VS endpoint + delta sync index (waits for ONLINE)
  Step 4: register_model         → Build wheel, log model, register in UC, set @champion alias
  Step 5: pre_deployment_eval    → Load via mlflow.pyfunc.load_model(), LLM-as-judge eval, quality gate
  Step 6: deploy_agent           → Deploy endpoint (champion/challenger traffic splits)
  Step 7: smoke_test             → Validate live endpoint (8 tests)
  Step 8: post_deployment_eval   → Evaluate live endpoint, LLM-as-judge scores

Mastercard Pipeline Status:
  ✅ Step 1-4: PASSING
  ✅ Step 5 (pre_deployment_eval): PASSING — LLM-as-judge scores 5/5
  ✅ Step 6 (deploy_agent): PASSING
  ✅ Step 7 (smoke_test): PASSING
  🔄 Step 8 (post_deployment_eval): In progress (pip install + mlflow.genai fix applied)


Evaluation Architecture (mlflow.genai.evaluate):
─────────────────────────────────────────────────
API: mlflow.genai.evaluate() with @scorer functions
Judge Model: databricks-meta-llama-3-3-70b-instruct (internal FMAPI, no internet)

7 LLM-as-judge scorers:
  accuracy (1-5)           — Factual correctness
  helpfulness (1-5)        — Actionable, practical guidance
  professionalism (1-5)    — Formal tone
  docs_relevance (1-5)     — Databricks-specific content
  code_snippet_quality (1-5) — Code examples for coding questions
  source_citation (1-5)    — References to documentation
  answer_completeness (1-5) — Thorough, non-deflecting answers

Scorer implementation: Each @scorer calls _call_judge() which queries the LLM
endpoint via Databricks SDK. Response parsed with regex to extract 1-5 score.

MLflow version compatibility:
  - MLflow 3.3.2 (Mastercard): metrics use {name}/mean, scores in assessments[].feedback.value
  - MLflow 3.10+ (FEVM): metrics use {name}/v1/mean, scores in {name}/value columns
  - Code handles both formats automatically

Key fix: pass only ["inputs", "outputs", "expectations"] to mlflow.genai.evaluate()
  - MLflow 3.x auto-maps "request" → "inputs" internally, overwriting dict column
  - inputs must be list comprehension [{"query": r} for r in series], NOT .apply(lambda)
  - pandas .apply() wraps dicts in a way that fails isinstance(x, dict) in MLflow 3.3.2


Air-Gapped / Mastercard-Specific:
─────────────────────────────────
  Pip install:
    - Conditional: auto-detects Mastercard volume → falls back to PyPI
    - Volume: /Volumes/mc_edacde_shared/datalake_shared/libraries/dip/enc/python/312/
    - --no-index flag prevents all PyPI calls
    - --no-build-isolation for wheel build (uses cluster's setuptools)
    - All notebooks with pip install also have restartPython() + widget re-read

  Data:
    - Bundled dataset: 4568 docs in datasets/databricks_docs.json (14 MB)
    - data_source_url: "local" in Mastercard target reads from bundled JSON
    - No internet needed for data ingestion

  Evaluation:
    - mlflow.genai.evaluate() — no default evaluator, no tiktoken/HuggingFace downloads
    - @scorer functions call internal Databricks FMAPI (no internet)
    - Zero PyPI calls in production

  Vector Search:
    - Uses Databricks SDK only (w.vector_search_indexes.query_index)
    - No databricks-vectorsearch package needed
    - get_or_create_endpoint waits for ONLINE (600s)
    - create_delta_sync_index waits for sync complete (1200s)
    - Column names: chunk_text, url, chunk_id (from chunked table)

  Wheel Build:
    - Framework wheel embedded in model artifact (conda_env + artifacts pattern)
    - UC Volume stores wheel for versioning/audit
    - No external dependencies in pyproject.toml (all pre-installed on ML runtime)

  Utilities:
    - mlops_utils.py integrated: retry, checkpoint, volume staging, SHA-256 verify
    - SHARED_LIBS_BASE paths for Mastercard Python libraries


Project Structure:
──────────────────
AgentOPS/
├── agentops_demo/
│   ├── databricks.yml                          # Variables, targets (dev/mastercard/stage)
│   ├── pyproject.toml                          # Framework wheel (v1.6.0, no deps)
│   │
│   ├── framework/                              # Standardized core (→ wheel artifact)
│   │   ├── agent_base.py                       # AgentOPSBase(ChatAgent)
│   │   ├── mlops_utils.py                      # Mastercard MLOps utilities
│   │   ├── guardrails/{pre_llm,post_llm}.py    # 6 pre + 5 post LLM checks
│   │   ├── evaluation/evaluation_pipeline.py   # run_evaluation() via mlflow.genai.evaluate()
│   │   ├── audit/audit_logger.py               # PipelineStepLogger + _safe_json_dumps
│   │   ├── monitoring/trace_monitor.py         # Metrics, guardrail stats, drift
│   │   ├── optimization/prompt_optimizer.py    # MemAlign + GEPA
│   │   └── inference/batch_runner.py           # ai_query() with quarantine
│   │
│   ├── data_preparation/
│   │   ├── data_ingestion/
│   │   │   ├── datasets/databricks_docs.json   # 4568 docs bundled (14 MB)
│   │   │   ├── ingestion/fetch_data.py         # fetch_data_from_url + load_data_from_file
│   │   │   └── notebooks/DataIngestion.py
│   │   ├── data_preprocessing/
│   │   │   └── notebooks/DataPreprocessing.py  # → chunk_text, url, chunk_id
│   │   └── vector_search/
│   │       ├── vector_search_utils/utils.py    # SDK only, waits for ONLINE
│   │       └── notebooks/VectorSearch.py
│   │
│   ├── agent_development/
│   │   ├── agent/
│   │   │   ├── notebooks/Agent.py              # DatabricksDocsAgent (RAG)
│   │   │   ├── notebooks/RegisterModel.py      # Log + register (no deploy)
│   │   │   ├── notebooks/DeployAgent.py        # Deploy endpoint (champion/challenger)
│   │   │   ├── config.yaml                     # Agent config (merged at deploy)
│   │   │   └── tools/agent_tools.py            # search_docs (SDK)
│   │   └── agent_evaluation/
│   │       ├── evaluation/custom_scorers.py    # 7 @scorer LLM-as-judge functions
│   │       ├── evaluation/{golden,adversarial}_dataset.json
│   │       └── notebooks/
│   │           ├── PreDeploymentEval.py         # pyfunc.load_model + genai.evaluate
│   │           ├── PostDeploymentEval.py        # Live endpoint + genai.evaluate
│   │           ├── SmokeTest.py                 # 8 endpoint tests
│   │           ├── RunMonitoring.py             # Hourly + Lakehouse Monitor
│   │           ├── IterativeImprovement.py      # MemAlign + GEPA
│   │           └── AggregateAudit.py            # Daily guardrail summary
│   │
│   ├── agent_deployment/
│   │   ├── model_serving/notebooks/UpdateTraffic.py
│   │   ├── batch_inference/notebooks/RunBatchInference.py
│   │   └── monitoring/notebooks/SetupLakehouseMonitoring.py
│   │
│   └── resources/
│       ├── pipeline-resource.yml               # 8-step pipeline
│       ├── monitoring-resource.yml             # Hourly + daily jobs
│       ├── batch-inference-resource.yml        # Daily batch job
│       └── serving-resource.yml.reference      # AI Gateway config


Config Architecture:
────────────────────
  databricks.yml → Infrastructure: workspaces, catalogs, endpoints, cluster_id
  config.yaml    → Agent behavior: LLM, prompt, guardrails, history turns
  RegisterModel.py merges both → runtime_config.yaml baked into model artifact

  Key: databricks.yml is single source of truth for infra settings.
  config.yaml values are overridden by databricks.yml at deploy time.


Cluster Configuration:
──────────────────────
  - Mastercard: policy_id mc_edacde_bi_personal_compute, DBR 17.3 ML
  - FEVM: existing cluster via cluster_id variable
  - All jobs use existing_cluster_id: ${var.cluster_id}


Remaining:
──────────
  ✅ Pre-deployment eval — PASSING on Mastercard (scores 5/5)
  🔄 Post-deployment eval — pip install fix applied, testing
  ⏳ 3 scheduled jobs — monitoring, audit aggregation, batch inference
  ⏳ Lakehouse monitoring setup
  ⏳ CI/CD Jenkinsfile
  ⏳ Documentation update

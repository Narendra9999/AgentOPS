# AgentOPS — Agent Operations Framework

Production framework for deploying AI agents on Databricks with guardrails, evaluation, monitoring, and CI/CD.

## Quick Start

```bash
# 1. Configure
vim databricks.yml    # Set workspace URLs, catalogs, LLM endpoint

# 2. Test locally
python -m pytest tests/unit/ -v

# 3. Deploy
databricks bundle validate -t dev
databricks bundle deploy -t dev

# 4. Run the pipeline (one command — does everything)
databricks bundle run agentops_pipeline -t dev
```

The pipeline runs 8 chained steps — each tracked in audit tables:
```
data_ingestion → data_preprocessing → vector_search_setup → register_model
  → pre_deployment_eval → deploy_agent → smoke_test → post_deployment_eval
```

See [docs/agentops-setup.md](docs/agentops-setup.md) for the full setup guide.

## What's Included

| Component | Description |
|-----------|-------------|
| **Pipeline** | 8-step chained job: data prep → model → eval → endpoint → smoke test → post-eval |
| **Guardrails** | Pre/post LLM safety: PII, injection, toxicity, intent + AI Gateway safety filter |
| **Evaluation** | 7 LLM-as-judge scorers via mlflow.genai.evaluate(), quality gates, UC results table |
| **Audit** | Every pipeline step tracked with timing, status, records processed |
| **Monitoring** | Hourly trace metrics, drift detection, Lakehouse dashboards |
| **Iterative Improvement** | Expert feedback → MemAlign → GEPA prompt optimization |
| **Batch Inference** | Daily ai_query() with quarantine for blocked records |
| **Champion/Challenger** | A/B testing with traffic routing via AI Gateway |

## Resources Deployed

| Resource | Schedule | Tasks |
|----------|----------|-------|
| `agentops-pipeline` | Manual / CI/CD | 8 chained: ingestion → preprocessing → VS → register → pre-eval → deploy → smoke → post-eval |
| `agentops-monitoring` | Hourly | Trace metrics + guardrail stats + drift |
| `agentops-audit-aggregation` | Daily | Summarize guardrail events |
| `agentops-batch-inference` | Daily 6am | Batch queries through endpoint |

## Framework Version

**2.0.1** — `src/` layout, DAB-aligned, air-gapped compatible

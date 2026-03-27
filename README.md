# AgentOPS — Agent Operations Framework

Production framework for deploying AI agents on Databricks with guardrails, evaluation, monitoring, and CI/CD.

## Quick Start

```bash
# 1. Configure
vim agentops_demo/databricks.yml    # Set workspace URLs, catalogs, LLM endpoint

# 2. Test locally
python -m pytest tests/unit/ -v     # 51 tests

# 3. Deploy
databricks bundle validate -t dev
databricks bundle deploy -t dev

# 4. Run the pipeline (one command — does everything)
databricks bundle run agentops_pipeline -t dev
```

The pipeline runs 6 chained steps — each tracked in audit tables:
```
data_ingestion → data_preprocessing → vector_search → register_model → create_endpoint → evaluation
```

See [docs/agentops-setup.md](docs/agentops-setup.md) for the full setup guide.

## What's Included

| Component | Description |
|-----------|-------------|
| **Pipeline** | 6-step chained job: data prep → model → endpoint → eval |
| **Guardrails** | Pre/post LLM safety: PII, injection, toxicity, LlamaGuard, intent |
| **Evaluation** | MLflow quality gates + per-row results in UC tables |
| **Audit** | Every pipeline step tracked with timing, status, records |
| **Monitoring** | Hourly trace metrics, drift detection, Lakehouse dashboards |
| **Iterative Improvement** | Expert feedback → MemAlign → GEPA prompt optimization |
| **Batch Inference** | Daily ai_query() with quarantine for blocked records |
| **Champion/Challenger** | A/B testing with traffic routing |

## Resources Deployed

| Resource | Schedule | Tasks |
|----------|----------|-------|
| `agentops-pipeline` | Manual / CI/CD | 6 chained: ingestion → preprocessing → VS → register → endpoint → eval |
| `agentops-monitoring` | Hourly | Trace metrics + guardrail stats + drift |
| `agentops-audit-aggregation` | Daily | Summarize guardrail events |
| `agentops-batch-inference` | Daily 6am | Batch queries through endpoint |

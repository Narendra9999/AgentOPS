# Team Configurations

Each team deploys their own agent via PR. The directory structure:

```
teams/<your-team>/
  target.yml              # DAB config + Mastercard Jenkins fields + variables
  config.yaml             # Agent: system prompt, guardrails, eval settings
  fixtures/
    golden_dataset.json   # Eval questions + expected answers
    adversarial_dataset.json  # Guardrail test cases
  scorers/
    domain/*.yaml         # Domain-specific evaluation rules
    llm_judge/*.yaml      # LLM-as-judge rubrics
```

## Adding a New Team

1. Copy the template:
   ```bash
   cp -r teams/_template teams/<your-team>
   ```

2. Edit `target.yml`:
   - Fill in Mastercard DAB config (`sp_id`, `alert_mail_id`, `workspace_name`)
   - Set DAB variables (`catalog`, `schema`, `cluster_id`, `agent_name`, `chatbot_name`)
   - Set `team_dir` and `team_config` to your team folder name

3. Edit `config.yaml`:
   - Write your system prompt (domain-specific instructions)
   - Configure guardrail rules and blocked phrases
   - Set evaluation thresholds

4. Add evaluation data:
   - `fixtures/golden_dataset.json` — 10+ Q&A pairs for quality evaluation
   - `fixtures/adversarial_dataset.json` — guardrail test cases (injection, PII, etc.)

5. Add custom scorers (optional):
   - `scorers/domain/*.yaml` — guidelines rules for your domain
   - `scorers/llm_judge/*.yaml` — rubric-based LLM evaluation criteria

6. Submit PR — Jenkins triggers on merge:
   ```
   databricks bundle deploy -t team-<your-team>
   databricks bundle run agentops_pipeline -t team-<your-team>
   ```

## How It Works

```
PR merged → Jenkins reads target.yml
  → databricks bundle deploy -t team-<name>
    → uploads all code + team config to workspace
  → databricks bundle run agentops_pipeline -t team-<name>
    → Step 1: Data Ingestion (team fixtures or sitemap)
    → Step 2: Data Preprocessing (chunking + embeddings)
    → Step 3: Vector Search Setup (team-specific index)
    → Step 4: Register Model (team config.yaml → system prompt, guardrails)
    → Step 5: Pre-Deployment Eval (team fixtures + scorers)
    → Step 6: Deploy Agent (team-specific serving endpoint)
    → Step 7: Smoke Test
    → Step 8: Post-Deployment Eval (team fixtures + scorers)
```

## Team Isolation

Each team gets isolated resources — no cross-team interference:

| Resource | Naming | Example |
|----------|--------|---------|
| Schema | `{team}_agent` | `platform_eng_agent` |
| Audit schema | `{team}_audit` | `platform_eng_audit` |
| Vector search index | `{team}_docs_index` | `platform_eng_docs_index` |
| Serving endpoint | `{team}-chatbot` | `platform-eng-chatbot` |
| Model | `{team}_docs_agent` | `platform_eng_docs_agent` |

## What Teams Customize vs What the Framework Provides

| Teams customize | Framework provides |
|----------------|-------------------|
| System prompt | RAG pipeline (ingestion, chunking, VS) |
| Guardrail rules | Pre/post LLM guardrails engine |
| Evaluation scorers + datasets | LLM-as-judge evaluation framework |
| LLM endpoint choice | Model serving + endpoint management |
| Domain-specific blocked phrases | MLflow tracing + audit logging |

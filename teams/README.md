# Team Configurations

Each team deploys their own agent via PR. The directory structure:

```
teams/<your-team>/
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

2. Edit `config.yaml` — system prompt, guardrails, evaluation settings

3. Add evaluation data:
   - `fixtures/golden_dataset.json` — 10+ Q&A pairs for quality evaluation
   - `fixtures/adversarial_dataset.json` — guardrail test cases

4. Add custom scorers (optional):
   - `scorers/domain/*.yaml` — guidelines rules
   - `scorers/llm_judge/*.yaml` — rubric-based LLM evaluation

5. Update `dab-config.yml` (project root) with your team variables:
   ```yaml
   variables:
     schema: "my_team_agent"
     audit_schema: "my_team_audit"
     agent_name: "my_team_docs_agent"
     chatbot_name: "my-team-chatbot"
     vs_index: "my_team_docs_index"
     team_dir: "my-team"
     team_config: "teams/my-team/config.yaml"
   ```

6. Submit PR — Jenkins deploys:
   ```
   databricks bundle deploy -t work --var="schema=my_team_agent" --var="agent_name=..." ...
   databricks bundle run agentops_pipeline -t work
   ```

## How It Works

```
Team submits PR → updates teams/<team>/ + dab-config.yml variables
  → Jenkins reads dab-config.yml
  → databricks bundle deploy -t work --var=... (team variables)
  → databricks bundle run agentops_pipeline -t work
    → Step 1: Data Ingestion (team fixtures via team_dir)
    → Step 2: Data Preprocessing
    → Step 3: Vector Search Setup (team vs_index)
    → Step 4: Register Model (team config.yaml → system prompt, guardrails)
    → Step 5: Pre-Deployment Eval (team fixtures + scorers)
    → Step 6: Deploy Agent (team chatbot_name → separate endpoint)
    → Step 7: Smoke Test
    → Step 8: Post-Deployment Eval
```

## Team Isolation

Each team gets isolated resources via variable overrides:

| Resource | Variable | Example |
|----------|----------|---------|
| Schema | `schema` | `platform_eng_agent` |
| Audit schema | `audit_schema` | `platform_eng_audit` |
| Vector search index | `vs_index` | `platform_eng_docs_index` |
| Serving endpoint | `chatbot_name` | `platform-eng-chatbot` |
| Model | `agent_name` | `platform_eng_docs_agent` |
| Agent config | `team_config` | `teams/platform-engineering/config.yaml` |
| Eval fixtures | `team_dir` | `platform-engineering` |

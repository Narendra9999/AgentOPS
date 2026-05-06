# Team Configurations

Each team gets their own directory with:
- `target.yml` — Databricks bundle target configuration (workspace, catalog, variables)
- `config.yaml` — Agent configuration (system prompt, guardrails, evaluation scorers)
- `fixtures/` — (optional) Team-specific documentation data for air-gapped environments

## Adding a New Team

1. Copy the `_template/` directory:
   ```bash
   cp -r teams/_template teams/my-team
   ```

2. Edit `teams/my-team/target.yml` — set workspace, catalog, cluster, endpoints

3. Edit `teams/my-team/config.yaml` — customize system prompt, guardrails, scorers

4. (Optional) Add team-specific docs to `teams/my-team/fixtures/`

5. Submit a PR — once merged, Jenkins deploys automatically

## How It Works

- `databricks.yml` includes `teams/*/target.yml` via the include mechanism
- Each team's target is named: `team-<team-name>` (e.g., `team-hr`, `team-compliance`)
- Jenkins runs: `databricks bundle deploy -t team-<name> && databricks bundle run agentops_pipeline -t team-<name>`
- Each team gets their own: schema, vector search index, model, serving endpoint

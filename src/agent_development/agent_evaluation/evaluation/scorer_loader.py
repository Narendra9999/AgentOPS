"""
Scorer Loader — Builds the list of scorers based on config.yaml evaluation settings.

Supports 3 scorer types that can be used individually or combined:
  - builtin:    MLflow built-in scorers (Guidelines, Correctness, Safety, etc.)
  - llm_judge:  Custom LLM-as-judge scorers with detailed rubrics (accuracy, helpfulness, etc.)
  - domain:     Domain-specific scorers defined in config.yaml
  - all:        All three types combined

Default: "builtin" (no external judge calls needed, safest for air-gapped environments)

Customization per scorer:
  - enabled:   Toggle individual scorers on/off without removing config
  - weight:    Relative importance for weighted quality gate (default 1.0)
  - threshold: Per-scorer override of quality_gate_threshold (null = use global)

Parallel evaluation (mode="all"):
  - Scorer groups (builtin, llm_judge, domain) run concurrently via ThreadPoolExecutor
  - Each group is a separate mlflow.genai.evaluate() call with its own MLflow trace span
  - Results are merged into a single metrics dict
"""

import logging
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────

def _parse_scorer_modes(mode: str) -> set:
    """Parse scorer mode string into a set of active modes.

    Supports: "builtin", "llm_judge", "domain", "all",
    or comma-separated combinations like "builtin,domain".
    """
    if mode == "all":
        return {"builtin", "llm_judge", "domain"}
    return {m.strip() for m in mode.split(",") if m.strip()}


def load_scorers(eval_config: dict, judge_model: str = None) -> List:
    """
    Load scorers based on the evaluation config section of config.yaml.

    Args:
        eval_config: The 'evaluation' section from config.yaml
        judge_model: Override judge model endpoint name

    Returns:
        List of scorer objects ready for mlflow.genai.evaluate(scorers=...)
    """
    modes = _parse_scorer_modes(eval_config.get("scorer_mode", "builtin"))
    judge = judge_model or eval_config.get("judge_model", "databricks-meta-llama-3-3-70b-instruct")

    scorers = []

    if "builtin" in modes:
        scorers.extend(_load_builtin_scorers(eval_config.get("builtin_scorers", {}), judge))

    if "llm_judge" in modes:
        scorers.extend(_load_llm_judge_scorers(eval_config.get("llm_judge_scorers", {})))

    if "domain" in modes:
        scorers.extend(_load_domain_scorers(eval_config.get("domain_scorers", []), judge))

    logger.info(f"Loaded {len(scorers)} scorers (modes={modes}): {[_scorer_name(s) for s in scorers]}")
    return scorers


def load_scorer_groups(eval_config: dict, judge_model: str = None) -> Dict[str, List]:
    """
    Load scorers grouped by type for parallel execution.

    Supports any combination of modes via comma-separated string:
      "builtin"           → 1 group
      "builtin,domain"    → 2 groups (parallel)
      "all"               → 3 groups (parallel)

    Returns dict of {group_name: [scorers]} where each group can be
    evaluated independently in parallel.
    """
    modes = _parse_scorer_modes(eval_config.get("scorer_mode", "builtin"))
    judge = judge_model or eval_config.get("judge_model", "databricks-meta-llama-3-3-70b-instruct")

    groups = {}

    if "builtin" in modes:
        builtin = _load_builtin_scorers(eval_config.get("builtin_scorers", {}), judge)
        if builtin:
            groups["builtin"] = builtin

    if "llm_judge" in modes:
        llm_judge = _load_llm_judge_scorers(eval_config.get("llm_judge_scorers", {}))
        if llm_judge:
            groups["llm_judge"] = llm_judge

    if "domain" in modes:
        domain = _load_domain_scorers(eval_config.get("domain_scorers", []), judge)
        if domain:
            groups["domain"] = domain

    total = sum(len(s) for s in groups.values())
    logger.info(f"Loaded {total} scorers in {len(groups)} groups (modes={modes}): "
                f"{{{', '.join(f'{k}: {len(v)}' for k, v in groups.items())}}}")
    return groups


def run_parallel_evaluation(eval_data, scorer_groups: Dict[str, List]) -> dict:
    """
    Run scorer groups in parallel, each with its own MLflow trace span.

    Each group runs as a separate mlflow.genai.evaluate() call in a thread.
    MLflow traces each group as a child span under the parent evaluation span.

    Args:
        eval_data: DataFrame with inputs, outputs, expectations columns
        scorer_groups: Dict from load_scorer_groups()

    Returns:
        dict with merged metrics, per-group results, and timing info
    """
    import mlflow
    import time

    if len(scorer_groups) <= 1:
        # Single group — no parallelism needed
        group_name = next(iter(scorer_groups), "default")
        scorers = next(iter(scorer_groups.values()), [])
        result = mlflow.genai.evaluate(data=eval_data, scorers=scorers)
        return {
            "metrics": result.metrics if result.metrics else {},
            "tables": result.tables if hasattr(result, "tables") else {},
            "group_results": {group_name: {"metrics": result.metrics, "duration_ms": 0}},
        }

    merged_metrics = {}
    merged_tables = {}
    group_results = {}
    start_time = time.time()

    @mlflow.trace(name="parallel_scorer_evaluation", span_type="EVALUATION")
    def _run_all_groups():
        nonlocal merged_metrics, merged_tables, group_results

        def _evaluate_group(group_name, scorers):
            """Run one scorer group with its own traced span."""
            @mlflow.trace(name=f"eval_group_{group_name}", span_type="EVALUATION")
            def _traced_eval():
                t0 = time.time()
                result = mlflow.genai.evaluate(data=eval_data, scorers=scorers)
                duration_ms = int((time.time() - t0) * 1000)
                scorer_names = [_scorer_name(s) for s in scorers]
                logger.info(f"Group '{group_name}' ({len(scorers)} scorers) completed "
                            f"in {duration_ms}ms: {scorer_names}")
                return {
                    "metrics": result.metrics if result.metrics else {},
                    "tables": result.tables if hasattr(result, "tables") else {},
                    "duration_ms": duration_ms,
                    "scorers": scorer_names,
                }
            return _traced_eval()

        with ThreadPoolExecutor(max_workers=len(scorer_groups)) as pool:
            futures = {
                pool.submit(_evaluate_group, name, scorers): name
                for name, scorers in scorer_groups.items()
            }
            for future in as_completed(futures):
                group_name = futures[future]
                try:
                    result = future.result()
                    group_results[group_name] = result
                    merged_metrics.update(result["metrics"])
                    for table_name, table_df in result.get("tables", {}).items():
                        if table_name not in merged_tables:
                            merged_tables[table_name] = table_df
                except Exception as e:
                    logger.error(f"Scorer group '{group_name}' failed: {e}")
                    group_results[group_name] = {"error": str(e), "duration_ms": 0}

    _run_all_groups()

    total_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Parallel evaluation complete: {len(scorer_groups)} groups, "
                f"{sum(len(s) for s in scorer_groups.values())} scorers, {total_ms}ms total")

    return {
        "metrics": merged_metrics,
        "tables": merged_tables,
        "group_results": group_results,
        "total_duration_ms": total_ms,
    }


def get_thresholds(eval_config: dict) -> dict:
    """
    Build quality gate thresholds for all active scorer types.

    Per-scorer threshold overrides the global quality_gate_threshold.
    Returns dict of {metric_name: threshold} for the quality gate.
    """
    modes = _parse_scorer_modes(eval_config.get("scorer_mode", "builtin"))
    global_threshold = eval_config.get("quality_gate_threshold", 3.5)

    thresholds = {}

    if "llm_judge" in modes:
        llm_cfg = eval_config.get("llm_judge_scorers", {})
        for name, cfg in llm_cfg.items():
            if cfg.get("enabled", True):
                t = cfg.get("threshold") or global_threshold
                thresholds[f"{name}/mean"] = t

    if "domain" in modes:
        for domain_cfg in eval_config.get("domain_scorers", []):
            if not domain_cfg.get("enabled", True):
                continue
            name = domain_cfg.get("name", "unnamed")
            scorer_type = domain_cfg.get("type", "guidelines")
            t = domain_cfg.get("threshold") or global_threshold
            # Rubric scorers produce numeric 1-5 scores → use mean threshold
            if scorer_type == "rubric":
                thresholds[f"{name}/mean"] = t
            # Guidelines/keyword/expected_facts produce pass/fail → use pass rate
            else:
                thresholds[f"{name}/average"] = t

    # Built-in scorers (Guidelines, Correctness, etc.) produce pass/fail.
    # Their aggregate metrics are pass rates (0.0-1.0).
    # We don't enforce hard thresholds on these by default — they are informational.
    # Users can add explicit thresholds via per-scorer config if needed.

    return thresholds


def get_weights(eval_config: dict) -> Dict[str, float]:
    """
    Build weight map for weighted quality gate scoring.

    Returns dict of {scorer_name: weight} for all active scorers.
    Useful for computing a weighted average score across scorers.
    """
    modes = _parse_scorer_modes(eval_config.get("scorer_mode", "builtin"))
    weights = {}

    if "llm_judge" in modes:
        for name, cfg in eval_config.get("llm_judge_scorers", {}).items():
            if cfg.get("enabled", True):
                weights[name] = cfg.get("weight", 1.0)

    if "domain" in modes:
        for domain_cfg in eval_config.get("domain_scorers", []):
            if domain_cfg.get("enabled", True):
                weights[domain_cfg.get("name", "unnamed")] = domain_cfg.get("weight", 1.0)

    if "builtin" in modes:
        for name, cfg in eval_config.get("builtin_scorers", {}).items():
            if cfg.get("enabled", True):
                weights[name] = cfg.get("weight", 1.0)

    return weights


# ── Internal helpers ────────────────────────────────────────────────────────

def _scorer_name(scorer) -> str:
    """Extract name from a scorer for logging."""
    if hasattr(scorer, "name"):
        return scorer.name
    if hasattr(scorer, "__name__"):
        return scorer.__name__
    return type(scorer).__name__


def _load_builtin_scorers(config: dict, judge_model: str) -> List:
    """Load MLflow built-in scorers based on config."""
    from mlflow.genai.scorers import (
        Guidelines,
        Correctness,
        RelevanceToQuery,
        Safety,
    )

    scorers = []
    model_uri = f"databricks:/{judge_model}"

    # Guidelines scorer
    guidelines_cfg = config.get("guidelines", {})
    if guidelines_cfg.get("enabled", True):
        rules = guidelines_cfg.get("rules", [
            "The response must directly address the user's question",
            "The response must not include fabricated information",
        ])
        scorers.append(Guidelines(
            name=guidelines_cfg.get("name", "response_quality"),
            guidelines=rules,
            model=model_uri,
        ))
        logger.info(f"  Loaded builtin: Guidelines ({len(rules)} rules)")

    # Correctness scorer
    if config.get("correctness", {}).get("enabled", True):
        scorers.append(Correctness(model=model_uri))
        logger.info("  Loaded builtin: Correctness")

    # RelevanceToQuery scorer
    if config.get("relevance_to_query", {}).get("enabled", True):
        scorers.append(RelevanceToQuery(model=model_uri))
        logger.info("  Loaded builtin: RelevanceToQuery")

    # Safety scorer
    if config.get("safety", {}).get("enabled", True):
        scorers.append(Safety(model=model_uri))
        logger.info("  Loaded builtin: Safety")

    # RetrievalGroundedness — only if enabled (requires RETRIEVER span)
    if config.get("retrieval_groundedness", {}).get("enabled", False):
        from mlflow.genai.scorers import RetrievalGroundedness
        scorers.append(RetrievalGroundedness(model=model_uri))
        logger.info("  Loaded builtin: RetrievalGroundedness")

    return scorers


def _load_llm_judge_scorers(config: dict) -> List:
    """Load custom LLM-as-judge scorers from YAML files or inline config.

    Looks for scorer definitions in:
      1. scorers/llm_judge/*.yaml (file-based, preferred)
      2. config.yaml llm_judge_scorers section (inline, fallback)
    """
    import yaml
    import os

    scorers_dir = os.path.join(os.path.dirname(__file__), "scorers", "llm_judge")
    scorer_configs = {}

    # Load from YAML files first
    if os.path.isdir(scorers_dir):
        for fname in sorted(os.listdir(scorers_dir)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(scorers_dir, fname)
                with open(fpath) as f:
                    cfg = yaml.safe_load(f) or {}
                name = cfg.get("name", fname.replace(".yaml", "").replace(".yml", ""))
                scorer_configs[name] = cfg
                logger.info(f"  Loaded llm_judge config from file: {fname}")

    # Merge with inline config (inline overrides file-based enabled/weight/threshold)
    for name, inline_cfg in config.items():
        if name in scorer_configs:
            # File exists — inline overrides only enabled/weight/threshold
            for key in ("enabled", "weight", "threshold"):
                if key in inline_cfg:
                    scorer_configs[name][key] = inline_cfg[key]
        else:
            # No file — use inline config as-is
            scorer_configs[name] = inline_cfg

    # Build scorer functions from configs
    from agent_development.agent_evaluation.evaluation.custom_scorers import _judge_score
    from mlflow.genai.scorers import scorer as mlflow_scorer

    scorers = []
    for name, cfg in scorer_configs.items():
        if not cfg.get("enabled", True):
            logger.info(f"  Skipped llm_judge (disabled): {name}")
            continue

        criteria = cfg.get("criteria")
        rubric = cfg.get("rubric")

        if criteria:
            # File-based scorer with criteria + rubric
            def _make_scorer(n, c, r):
                @mlflow_scorer
                def file_based_scorer(inputs, outputs, expectations=None):
                    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
                    return _judge_score(question, str(outputs or ""), c, rubric=r)
                file_based_scorer.__name__ = n
                file_based_scorer.name = n
                return file_based_scorer

            scorers.append(_make_scorer(name, criteria, rubric))
        else:
            # Legacy inline scorer — import by name
            try:
                from agent_development.agent_evaluation.evaluation import custom_scorers
                scorer_fn = getattr(custom_scorers, name, None)
                if scorer_fn:
                    scorers.append(scorer_fn)
                else:
                    logger.warning(f"  No scorer function found for: {name}")
                    continue
            except (ImportError, AttributeError):
                logger.warning(f"  Could not import scorer: {name}")
                continue

        logger.info(f"  Loaded llm_judge: {name} (weight={cfg.get('weight', 1.0)})")

    return scorers


def _load_domain_scorers(config: list, judge_model: str) -> List:
    """
    Load domain-specific scorers from YAML files or inline config.

    Looks for scorer definitions in:
      1. scorers/domain/*.yaml (file-based, preferred)
      2. config.yaml domain_scorers list (inline, fallback)

    Supports 4 types:
      - guidelines:     LLM evaluates against natural language rules (pass/fail)
      - rubric:         LLM scores 1-5 against detailed criteria + rubric
      - keyword:        Fast regex/keyword check, no LLM call (pass/fail)
      - expected_facts: Check if specific facts appear in the response
    """
    import yaml
    import os

    scorers_dir = os.path.join(os.path.dirname(__file__), "scorers", "domain")
    scorer_configs = {}  # name → config dict

    # Load from YAML files first
    if os.path.isdir(scorers_dir):
        for fname in sorted(os.listdir(scorers_dir)):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(scorers_dir, fname)
                with open(fpath) as f:
                    cfg = yaml.safe_load(f) or {}
                name = cfg.get("name", fname.replace(".yaml", "").replace(".yml", ""))
                scorer_configs[name] = cfg
                logger.info(f"  Loaded domain config from file: {fname}")

    # Merge inline configs — override enabled/weight/threshold for existing,
    # add new scorers that don't have YAML files
    for inline_cfg in (config or []):
        name = inline_cfg.get("name", "")
        if not name:
            continue
        if name in scorer_configs:
            # File exists — inline overrides enabled/weight/threshold/rules
            for key in ("enabled", "weight", "threshold", "rules"):
                if key in inline_cfg:
                    scorer_configs[name][key] = inline_cfg[key]
            logger.info(f"  Overrode domain scorer from config: {name}")
        else:
            # No file — use inline config as-is
            scorer_configs[name] = inline_cfg

    all_configs = list(scorer_configs.values())

    scorers = []
    model_uri = f"databricks:/{judge_model}"

    for domain_cfg in all_configs:
        if not domain_cfg.get("enabled", True):
            logger.info(f"  Skipped domain scorer (disabled): {domain_cfg.get('name', 'unnamed')}")
            continue

        name = domain_cfg.get("name", "unnamed_domain_scorer")
        scorer_type = domain_cfg.get("type", "guidelines")
        weight = domain_cfg.get("weight", 1.0)

        if scorer_type == "guidelines":
            from mlflow.genai.scorers import Guidelines
            rules = domain_cfg.get("rules", [])
            if rules:
                scorers.append(Guidelines(
                    name=name,
                    guidelines=rules,
                    model=model_uri,
                ))
                logger.info(f"  Loaded domain (guidelines): {name} ({len(rules)} rules, weight={weight})")

        elif scorer_type == "rubric":
            scorers.append(_build_rubric_scorer(
                name=name,
                criteria=domain_cfg.get("criteria", ""),
                rubric=domain_cfg.get("rubric", ""),
                judge_model=judge_model,
            ))
            logger.info(f"  Loaded domain (rubric): {name} (weight={weight})")

        elif scorer_type == "keyword":
            scorers.append(_build_keyword_scorer(
                name=name,
                mode=domain_cfg.get("mode", "must_contain_any"),
                patterns=domain_cfg.get("patterns", []),
            ))
            logger.info(f"  Loaded domain (keyword): {name} ({len(domain_cfg.get('patterns', []))} patterns, weight={weight})")

        elif scorer_type == "expected_facts":
            scorers.append(_build_expected_facts_scorer(
                name=name,
                min_facts_present=domain_cfg.get("min_facts_present", 0.8),
            ))
            logger.info(f"  Loaded domain (expected_facts): {name} (weight={weight})")

    return scorers


def _build_rubric_scorer(name: str, criteria: str, rubric: str, judge_model: str):
    """Build a custom LLM-as-judge scorer with a domain-specific rubric."""
    from mlflow.genai.scorers import scorer

    @scorer
    def domain_rubric_scorer(inputs, outputs, expectations=None):
        from agent_development.agent_evaluation.evaluation.custom_scorers import _judge_score
        question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
        return _judge_score(question, str(outputs or ""), criteria, rubric=rubric)

    domain_rubric_scorer.__name__ = name
    domain_rubric_scorer.name = name
    return domain_rubric_scorer


def _build_keyword_scorer(name: str, mode: str, patterns: list):
    """Build a fast keyword/regex scorer — no LLM call needed."""
    import re
    from mlflow.genai.scorers import scorer

    compiled = [re.compile(p) for p in patterns]

    @scorer
    def domain_keyword_scorer(inputs, outputs, expectations=None):
        text = str(outputs or "")
        matches = [bool(p.search(text)) for p in compiled]

        if mode == "must_contain_all":
            return "yes" if all(matches) else "no"
        elif mode == "must_contain_any":
            return "yes" if any(matches) else "no"
        elif mode == "must_not_contain":
            return "yes" if not any(matches) else "no"
        return "yes"

    domain_keyword_scorer.__name__ = name
    domain_keyword_scorer.name = name
    return domain_keyword_scorer


def _build_expected_facts_scorer(name: str, min_facts_present: float):
    """Build a scorer that checks if expected facts appear in the response."""
    from mlflow.genai.scorers import scorer

    @scorer
    def domain_facts_scorer(inputs, outputs, expectations=None):
        text = str(outputs or "").lower()
        if not expectations or not isinstance(expectations, dict):
            return "yes"  # No expected facts to check

        facts = expectations.get("expected_facts", [])
        if not facts:
            return "yes"

        found = sum(1 for fact in facts if fact.lower() in text)
        ratio = found / len(facts) if facts else 1.0
        return "yes" if ratio >= min_facts_present else "no"

    domain_facts_scorer.__name__ = name
    domain_facts_scorer.name = name
    return domain_facts_scorer

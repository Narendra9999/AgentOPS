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
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────

def load_scorers(eval_config: dict, judge_model: str = None) -> List:
    """
    Load scorers based on the evaluation config section of config.yaml.

    Args:
        eval_config: The 'evaluation' section from config.yaml
        judge_model: Override judge model endpoint name

    Returns:
        List of scorer objects ready for mlflow.genai.evaluate(scorers=...)
    """
    mode = eval_config.get("scorer_mode", "builtin")
    judge = judge_model or eval_config.get("judge_model", "databricks-meta-llama-3-3-70b-instruct")

    scorers = []

    if mode in ("builtin", "all"):
        scorers.extend(_load_builtin_scorers(eval_config.get("builtin_scorers", {}), judge))

    if mode in ("llm_judge", "all"):
        scorers.extend(_load_llm_judge_scorers(eval_config.get("llm_judge_scorers", {})))

    if mode in ("domain", "all"):
        scorers.extend(_load_domain_scorers(eval_config.get("domain_scorers", []), judge))

    logger.info(f"Loaded {len(scorers)} scorers (mode={mode}): {[_scorer_name(s) for s in scorers]}")
    return scorers


def get_thresholds(eval_config: dict) -> dict:
    """
    Build quality gate thresholds for all active scorer types.

    Per-scorer threshold overrides the global quality_gate_threshold.
    Returns dict of {metric_name: threshold} for the quality gate.
    """
    mode = eval_config.get("scorer_mode", "builtin")
    global_threshold = eval_config.get("quality_gate_threshold", 3.5)

    thresholds = {}

    if mode in ("llm_judge", "all"):
        llm_cfg = eval_config.get("llm_judge_scorers", {})
        for name, cfg in llm_cfg.items():
            if cfg.get("enabled", True):
                t = cfg.get("threshold") or global_threshold
                thresholds[f"{name}/mean"] = t

    if mode in ("domain", "all"):
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
    mode = eval_config.get("scorer_mode", "builtin")
    weights = {}

    if mode in ("llm_judge", "all"):
        for name, cfg in eval_config.get("llm_judge_scorers", {}).items():
            if cfg.get("enabled", True):
                weights[name] = cfg.get("weight", 1.0)

    if mode in ("domain", "all"):
        for domain_cfg in eval_config.get("domain_scorers", []):
            if domain_cfg.get("enabled", True):
                weights[domain_cfg.get("name", "unnamed")] = domain_cfg.get("weight", 1.0)

    if mode in ("builtin", "all"):
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
    """Load custom LLM-as-judge scorers based on config."""
    from agent_development.agent_evaluation.evaluation.custom_scorers import (
        accuracy, helpfulness, professionalism, docs_relevance,
        code_snippet_quality, source_citation, answer_completeness,
    )

    all_scorers = {
        "accuracy": accuracy,
        "helpfulness": helpfulness,
        "professionalism": professionalism,
        "docs_relevance": docs_relevance,
        "code_snippet_quality": code_snippet_quality,
        "source_citation": source_citation,
        "answer_completeness": answer_completeness,
    }

    scorers = []
    for name, scorer_fn in all_scorers.items():
        scorer_cfg = config.get(name, {})
        if scorer_cfg.get("enabled", True):
            scorers.append(scorer_fn)
            logger.info(f"  Loaded llm_judge: {name} (weight={scorer_cfg.get('weight', 1.0)})")

    return scorers


def _load_domain_scorers(config: list, judge_model: str) -> List:
    """
    Load domain-specific scorers from config.yaml definitions.

    Supports 4 types:
      - guidelines:     LLM evaluates against natural language rules (pass/fail)
      - rubric:         LLM scores 1-5 against detailed criteria + rubric
      - keyword:        Fast regex/keyword check, no LLM call (pass/fail)
      - expected_facts: Check if specific facts appear in the response
    """
    scorers = []
    model_uri = f"databricks:/{judge_model}"

    for domain_cfg in config:
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

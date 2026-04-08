"""
AgentOPS Framework — Evaluation Pipeline
Uses mlflow.evaluate() with model_type="question-answering" for built-in metrics
plus LLM-as-judge custom metrics (accuracy, helpfulness, professionalism).
Saves per-row results to UC audit table.
Used as CI/CD quality gate before environment promotion.
"""

import mlflow
import pandas as pd
import logging
import json
import uuid
import requests
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    # MLflow 3.3.2 uses {name}/mean, MLflow 3.10+ uses {name}/v1/mean
    "accuracy/mean": 3.5,
    "helpfulness/mean": 3.5,
    "professionalism/mean": 3.5,
    "docs_relevance/mean": 3.5,
    "answer_completeness/mean": 3.5,
}


def load_evaluation_dataset(path_or_table: str, spark=None) -> pd.DataFrame:
    """Load evaluation dataset from JSON file or UC table."""
    if path_or_table.endswith(".json"):
        return pd.read_json(path_or_table)
    if spark:
        return spark.table(path_or_table).toPandas()
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.getOrCreate()
    return spark.table(path_or_table).toPandas()


def _build_query_fn(endpoint_name: str):
    """Build a function that calls the serving endpoint for each eval row."""
    def query_iteration(inputs_df):
        token = os.environ.get("DATABRICKS_TOKEN", "")
        host = os.environ.get("DATABRICKS_HOST", "")
        if not host.startswith("http"):
            host = f"https://{host}"
        url = f"{host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        answers = []
        for _, row in inputs_df.iterrows():
            query = row["inputs"]
            try:
                resp = requests.post(url, headers=headers, json={
                    "messages": [{"role": "user", "content": query}]
                })
                resp.raise_for_status()
                data = resp.json()
                if "messages" in data and data["messages"]:
                    answers.append(data["messages"][0].get("content", ""))
                elif "choices" in data and data["choices"]:
                    answers.append(data["choices"][0]["message"]["content"])
                else:
                    answers.append(str(data))
            except Exception as e:
                logger.error(f"Endpoint query failed: {e}")
                answers.append(f"Error: {e}")
        return answers

    return query_iteration


def save_eval_results_to_table(
    spark,
    eval_result: dict,
    catalog: str,
    audit_schema: str,
    agent_name: str,
    agent_version: str = "1.0.0",
    environment: str = "dev",
    execution_id: str = "",
    results_table_name: str = "eval_results",
) -> str:
    """
    Save per-row evaluation results to the UC eval_results audit table.

    Uses the eval_results_table from mlflow.evaluate() which contains
    per-row scores from both built-in and LLM-as-judge metrics.
    """
    from pyspark.sql import functions as F

    evaluation_id = str(uuid.uuid4())

    per_row_df = eval_result.get("per_row_df")
    if per_row_df is None or per_row_df.empty:
        logger.warning("No per-row results to save")
        return evaluation_id

    # Log available columns for debugging
    if per_row_df is not None:
        score_cols = [c for c in per_row_df.columns if "/score" in c or "/v1" in c]
        logger.info(f"Available score columns: {score_cols}")

    def _get_score_from_assessments(row_dict, scorer_name):
        """Extract score from assessments field (MLflow 3.3.2 stores scores there)."""
        assessments = row_dict.get("assessments", [])
        if isinstance(assessments, list):
            for a in assessments:
                if isinstance(a, dict) and a.get("assessment_name") == scorer_name:
                    feedback = a.get("feedback", {})
                    if isinstance(feedback, dict):
                        val = feedback.get("value")
                        if val is not None:
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                pass
        return 0.0

    def _get_score(row_dict, scorer_name, *col_keys):
        """Extract score — try column names first, then assessments field."""
        # Try column-based extraction (MLflow 3.10+)
        for k in col_keys:
            v = row_dict.get(k)
            if v is not None and str(v) not in ("None", "nan", ""):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        # Fallback: extract from assessments (MLflow 3.3.2)
        return _get_score_from_assessments(row_dict, scorer_name)

    eval_rows = []
    for i, row in per_row_df.iterrows():
        row_dict = row.to_dict()
        eval_rows.append({
            "evaluation_id": evaluation_id,
            "execution_id": execution_id,
            "row_index": i,
            "request": (lambda r: r.get("query", str(r)) if isinstance(r, dict) else str(r))(row_dict.get("request", row_dict.get("inputs", "")))[:4000],
            "response": str(row_dict.get("response", row_dict.get("outputs", "")))[:4000],
            "expected_response": str(row_dict.get("targets", ""))[:4000],
            "context": "",
            "toxicity_score": _get_score(row_dict, "toxicity", "toxicity/value", "toxicity/v1/score"),
            "accuracy_score": _get_score(row_dict, "accuracy", "accuracy/value", "accuracy/v1/score"),
            "helpfulness_score": _get_score(row_dict, "helpfulness", "helpfulness/value", "helpfulness/v1/score"),
            "professionalism_score": _get_score(row_dict, "professionalism", "professionalism/value", "professionalism/v1/score"),
            "docs_relevance_score": _get_score(row_dict, "docs_relevance", "docs_relevance/value", "docs_relevance/v1/score"),
            "code_snippet_score": _get_score(row_dict, "code_snippet_quality", "code_snippet_quality/value", "code_snippet_quality/v1/score"),
            "source_citation_score": _get_score(row_dict, "source_citation", "source_citation/value", "source_citation/v1/score"),
            "answer_completeness_score": _get_score(row_dict, "answer_completeness", "answer_completeness/value", "answer_completeness/v1/score"),
            "overall_passed": bool(eval_result.get("passed", False)),
            "agent_name": agent_name,
            "agent_version": agent_version,
            "environment": environment,
        })

    table_name = f"{catalog}.{audit_schema}.{results_table_name}"
    df = spark.createDataFrame(eval_rows)
    df = df.withColumn("evaluated_at", F.current_timestamp())
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(table_name)

    logger.info(f"Saved {len(eval_rows)} eval results to {table_name} (evaluation_id={evaluation_id})")
    return evaluation_id


@mlflow.trace(span_type="EVALUATION")
def run_evaluation(
    eval_dataset: pd.DataFrame,
    scorers: list = None,
    scorer_groups: dict = None,
    thresholds: dict = None,
    model_endpoint: str = None,
) -> dict:
    """
    Run the standardized evaluation pipeline using mlflow.genai.evaluate().

    Supports two modes:
      - scorers: flat list → single sequential evaluate() call
      - scorer_groups: dict of {group: [scorers]} → parallel evaluate() calls
        Each group runs in its own thread with a dedicated MLflow trace span.

    Args:
        eval_dataset: DataFrame with 'request', 'expected_response' columns
        scorers: Flat list of scorer functions (sequential mode)
        scorer_groups: Dict of {group_name: [scorers]} from load_scorer_groups() (parallel mode)
        thresholds: Override default thresholds
        model_endpoint: Serving endpoint name to evaluate (for predict_fn)

    Returns:
        dict with pass/fail, metrics, gate results, per_row_df, and group timing
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    # Prepare dataset for mlflow.genai.evaluate()
    eval_df = eval_dataset.copy()
    if "inputs" not in eval_df.columns and "request" in eval_df.columns:
        eval_df["inputs"] = [{"query": r} for r in eval_df["request"]]
    if "expectations" not in eval_df.columns and "expected_response" in eval_df.columns:
        eval_df["expectations"] = [{"expected_response": r} for r in eval_df["expected_response"]]

    # Generate predictions from endpoint if provided
    if model_endpoint and "outputs" not in eval_df.columns:
        logger.info(f"Generating predictions from endpoint: {model_endpoint}")
        outputs = []
        for _, row in eval_df.iterrows():
            query = row["inputs"]["query"] if isinstance(row["inputs"], dict) else str(row["inputs"])
            try:
                token = os.environ.get("DATABRICKS_TOKEN", "")
                host = os.environ.get("DATABRICKS_HOST", "")
                if not host.startswith("http"):
                    host = f"https://{host}"
                resp = requests.post(
                    f"{host}/serving-endpoints/{model_endpoint}/invocations",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"messages": [{"role": "user", "content": query}]},
                )
                resp.raise_for_status()
                data = resp.json()
                if "messages" in data and data["messages"]:
                    outputs.append(data["messages"][0].get("content", ""))
                elif "choices" in data and data["choices"]:
                    outputs.append(data["choices"][0]["message"]["content"])
                else:
                    outputs.append(str(data))
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                outputs.append(f"Error: {e}")
        eval_df["outputs"] = outputs

    eval_data = eval_df[["inputs", "outputs", "expectations"]]

    # Run evaluation — parallel if scorer_groups provided, else sequential
    group_results = {}
    if scorer_groups and len(scorer_groups) > 1:
        from agent_development.agent_evaluation.evaluation.scorer_loader import run_parallel_evaluation
        parallel_result = run_parallel_evaluation(eval_data, scorer_groups)
        metrics = parallel_result["metrics"]
        tables = parallel_result.get("tables", {})
        group_results = parallel_result.get("group_results", {})
        logger.info(f"Parallel evaluation: {parallel_result.get('total_duration_ms')}ms total, "
                    f"groups: {list(group_results.keys())}")
    else:
        all_scorers = scorers or []
        if scorer_groups:
            all_scorers = [s for group in scorer_groups.values() for s in group]
        eval_result = mlflow.genai.evaluate(data=eval_data, scorers=all_scorers)
        metrics = eval_result.metrics if hasattr(eval_result, "metrics") and eval_result.metrics else {}
        tables = eval_result.tables if hasattr(eval_result, "tables") else {}

    logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2, default=str)}")

    # Check against thresholds
    gate_results = {}
    all_passed = True
    for metric_name, threshold in thresholds.items():
        actual = metrics.get(metric_name)
        if actual is not None:
            passed = actual >= threshold
            gate_results[metric_name] = {
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }
            if not passed:
                all_passed = False
                logger.warning(f"GATE FAILED: {metric_name} = {actual} (threshold: {threshold})")

    # Extract per-row results
    per_row_df = None
    if tables:
        per_row_df = tables.get("eval_results", tables.get("eval_results_table"))

    if per_row_df is not None:
        logger.info(f"Per-row df: shape={per_row_df.shape}, columns={list(per_row_df.columns)}")

    return {
        "passed": all_passed,
        "metrics": metrics,
        "gate_results": gate_results,
        "per_row_df": per_row_df,
        "group_results": group_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mlflow.trace(span_type="EVALUATION")
def run_guardrail_evaluation(agent, adversarial_dataset: pd.DataFrame) -> dict:
    """
    Run guardrail-specific evaluation against adversarial test suite.

    Args:
        agent: Instantiated agent with pre_llm_guardrails attribute
        adversarial_dataset: DataFrame with 'request', 'attack_type', 'should_block'

    Returns:
        dict with pass rates, false positive/negative rates, details
    """
    results = {
        "total": len(adversarial_dataset),
        "correct_blocks": 0,
        "correct_passes": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "details": [],
    }

    for _, row in adversarial_dataset.iterrows():
        input_text = row["request"]
        should_block = row["should_block"]

        pre_result = agent.pre_llm_guardrails.check(input_text)
        was_blocked = pre_result.get("blocked", False)

        if should_block and was_blocked:
            results["correct_blocks"] += 1
        elif not should_block and not was_blocked:
            results["correct_passes"] += 1
        elif should_block and not was_blocked:
            results["false_negatives"] += 1
        elif not should_block and was_blocked:
            results["false_positives"] += 1

        results["details"].append({
            "request": input_text[:100],
            "attack_type": row.get("attack_type", "unknown"),
            "should_block": should_block,
            "was_blocked": was_blocked,
            "correct": should_block == was_blocked,
            "blocked_by": pre_result.get("blocked_by", ""),
        })

    should_block_total = max(sum(1 for _, r in adversarial_dataset.iterrows() if r["should_block"]), 1)
    should_pass_total = max(sum(1 for _, r in adversarial_dataset.iterrows() if not r["should_block"]), 1)

    results["block_accuracy"] = results["correct_blocks"] / should_block_total
    results["pass_accuracy"] = results["correct_passes"] / should_pass_total
    results["false_positive_rate"] = results["false_positives"] / should_pass_total
    results["false_negative_rate"] = results["false_negatives"] / should_block_total
    results["overall_accuracy"] = (results["correct_blocks"] + results["correct_passes"]) / results["total"]

    return results

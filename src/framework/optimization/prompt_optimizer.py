"""
AgentOPS Framework — Prompt Optimization
Uses mlflow.genai.optimize_prompts() with GepaPromptOptimizer.

Requirements: mlflow>=3.5, databricks-sdk, dspy, openai
GEPA is built into MLflow — no separate gepa package needed.

Reference:
  https://docs.databricks.com/aws/en/mlflow3/genai/tutorials/examples/prompt-optimization-quickstart
  https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/automatically-optimize-prompts
"""

import mlflow
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def optimize_prompt(
    prompt_name: str,
    llm_endpoint: str,
    judge_model: str,
    eval_dataset: list[dict],
    max_metric_calls: int = 100,
    custom_scorers: list = None,
) -> dict:
    """
    Optimize a system prompt using GEPA via mlflow.genai.optimize_prompts().

    GEPA iteratively generates candidate prompt variations using LLM-driven
    reflection and automated feedback, then selects the best-performing prompt.

    Args:
        prompt_name: Fully qualified prompt name in MLflow registry (catalog.schema.name)
        llm_endpoint: Databricks model serving endpoint for the agent LLM
        judge_model: Model endpoint for GEPA reflection and scoring
        eval_dataset: List of dicts with 'inputs', 'outputs', 'expectations'
        max_metric_calls: Max evaluations during optimization
        custom_scorers: Additional scorers (defaults to Correctness + response_quality)

    Returns:
        dict with status, scores, and optimized prompt text
    """
    from mlflow.genai.optimize import GepaPromptOptimizer
    from mlflow.genai.scorers import Correctness
    from mlflow.genai.judges import make_judge
    from databricks_openai import DatabricksOpenAI

    reflection_model = f"databricks:/{judge_model}"
    scorer_model = f"databricks:/{judge_model}"
    openai_client = DatabricksOpenAI()

    # Load current prompt from registry
    try:
        current_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
        prompt_uri = current_prompt.uri
        logger.info(f"Loaded prompt: {prompt_name} (uri: {prompt_uri})")
    except Exception:
        logger.error(f"Prompt '{prompt_name}' not found in registry. Register it first.")
        return {"status": "failed", "error": f"Prompt '{prompt_name}' not found in registry"}

    # Define predict_fn — loads prompt from registry so GEPA can optimize it
    def predict_fn(question: str) -> str:
        loaded = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
        system_content = loaded.format()

        completion = openai_client.chat.completions.create(
            model=llm_endpoint,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": question},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        if isinstance(content, list):
            content = "".join(
                item.get("text", item.get("content", ""))
                for item in content if isinstance(item, dict)
            )
        return content

    # Define scorers
    scorers = custom_scorers or []
    if not scorers:
        scorers.append(Correctness(model=scorer_model))
        scorers.append(make_judge(
            name="response_quality",
            instructions=(
                "Evaluate the quality of the agent's response.\n\n"
                "Question: {{ inputs }}\n"
                "Response: {{ outputs }}\n\n"
                "Consider: accuracy, completeness, code quality, actionability."
            ),
            model=scorer_model,
        ))

    # Run GEPA optimization
    logger.info(f"Starting GEPA optimization (reflection: {reflection_model}, "
                f"max_calls: {max_metric_calls}, dataset: {len(eval_dataset)} examples)")

    import time
    t0 = time.time()

    try:
        result = mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=eval_dataset,
            prompt_uris=[prompt_uri],
            optimizer=GepaPromptOptimizer(
                reflection_model=reflection_model,
                max_metric_calls=max_metric_calls,
            ),
            scorers=scorers,
        )

        elapsed = time.time() - t0
        optimized_text = result.optimized_prompts[0].template

        logger.info(f"GEPA complete — {result.initial_eval_score:.3f} → "
                    f"{result.final_eval_score:.3f} in {elapsed:.0f}s")

        return {
            "status": "completed",
            "initial_score": float(result.initial_eval_score),
            "final_score": float(result.final_eval_score),
            "improvement": float(result.final_eval_score - result.initial_eval_score),
            "optimized_prompt": optimized_text,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"GEPA optimization failed after {elapsed:.0f}s: {e}")
        return {"status": "failed", "error": str(e), "elapsed_seconds": round(elapsed, 1)}


def register_optimized_prompt(
    prompt_name: str,
    optimized_text: str,
    initial_score: float,
    final_score: float,
    alias: str = "production",
) -> dict:
    """Register optimized prompt and update alias if improved."""
    if final_score <= initial_score:
        return {"status": "no_improvement", "initial": initial_score, "final": final_score}

    new_prompt = mlflow.genai.register_prompt(
        name=prompt_name,
        template=optimized_text,
        commit_message=f"GEPA optimized: {initial_score:.3f} → {final_score:.3f}",
        tags={"optimizer": "GEPA"},
    )

    mlflow.genai.set_prompt_alias(
        name=prompt_name,
        alias=alias,
        version=new_prompt.version,
    )

    logger.info(f"Registered {prompt_name} v{new_prompt.version}, @{alias} updated")
    return {"status": "registered", "version": new_prompt.version, "alias": alias}

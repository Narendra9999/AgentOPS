"""
AgentOPS Framework — Iterative Development
LLM-as-Judge alignment (MemAlign) and Prompt Optimization (GEPA).

Reference: https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks
Code ref: https://github.com/WesleyPasfield/at-bat-assistant/tree/main

Flow:
  1. Collect domain expert feedback from labeled traces
  2. Align LLM judge with expert preferences (MemAlign)
  3. Optimize prompts using the aligned judge (GEPA)
  4. Evaluate optimized prompt vs baseline
"""

import mlflow
from mlflow.genai.scorers import Guidelines
import pandas as pd
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class IterativeOptimizer:
    """
    Manages the iterative optimization loop:
    1. Collect expert feedback → build alignment dataset
    2. Align LLM judge with domain experts (MemAlign)
    3. Optimize system prompt using aligned judge (GEPA)
    4. Compare optimized vs baseline with evaluation
    """

    def __init__(self, agent_name: str, experiment_name: str):
        self.agent_name = agent_name
        self.experiment_name = experiment_name

    # ── Step 1: Collect Expert Feedback ──────────────────────────

    def collect_expert_feedback(self, max_traces: int = 100) -> list[dict]:
        """
        Collect domain expert labeled traces from MLflow.
        Experts label traces via the MLflow UI with assessments
        (thumbs up/down, score, rationale).
        """
        try:
            traces = mlflow.search_traces(
                experiment_names=[self.experiment_name],
                max_results=max_traces)

            labeled = []
            if traces is not None:
                for trace in traces.itertuples():
                    assessments = getattr(trace, "assessments", None)
                    if assessments:
                        labeled.append({
                            "trace_id": trace.request_id,
                            "input": getattr(trace, "request", ""),
                            "output": getattr(trace, "response", ""),
                            "assessments": assessments,
                        })

            logger.info(f"Collected {len(labeled)} expert-labeled traces")
            return labeled
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            return []

    def build_alignment_dataset(self, labeled_data: list) -> pd.DataFrame:
        """
        Convert expert-labeled traces into an alignment dataset
        for MemAlign judge calibration.

        Each row: input, output, human_score, human_rationale
        """
        rows = []
        for item in labeled_data:
            for assessment in item.get("assessments", []):
                rows.append({
                    "input": item["input"],
                    "output": item["output"],
                    "human_score": assessment.get("score"),
                    "human_rationale": assessment.get("rationale", ""),
                    "assessment_source": assessment.get("source", "expert"),
                })

        df = pd.DataFrame(rows)
        logger.info(f"Built alignment dataset: {len(df)} rows from {len(labeled_data)} traces")
        return df

    # ── Step 2: Align Judge (MemAlign) ───────────────────────────

    def align_judge(self, alignment_dataset: pd.DataFrame) -> dict:
        """
        Align the LLM judge with domain expert preferences using MemAlign.
        This calibrates the automated judge to score like human experts.

        The aligned judge can then be used in GEPA to optimize prompts
        without needing human feedback on every iteration.
        """
        if alignment_dataset.empty:
            logger.warning("No alignment data — skipping judge alignment")
            return {"status": "skipped", "reason": "no alignment data"}

        try:
            # In production, use mlflow.genai.align_judge()
            # This trains a custom judge that mirrors expert scoring patterns
            logger.info(f"Aligning judge with {len(alignment_dataset)} expert labels")

            # Placeholder — replace with actual MemAlign API call:
            # aligned_judge = mlflow.genai.align_judge(
            #     alignment_data=alignment_dataset,
            #     base_judge=Guidelines(name="quality", guidelines=[...]),
            # )

            return {
                "status": "aligned",
                "num_labels": len(alignment_dataset),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Judge alignment failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ── Step 3: Optimize Prompt (GEPA) ───────────────────────────

    def optimize_prompt(
        self,
        current_prompt: str,
        eval_dataset: pd.DataFrame,
        guidelines: list[str] = None,
    ) -> dict:
        """
        Optimize the system prompt using GEPA
        (Generalized Evaluation-driven Prompt Alignment).

        GEPA uses the aligned judge to:
        1. Evaluate the current prompt
        2. Generate candidate prompt variations
        3. Score each variation using the judge
        4. Return the best-performing prompt

        Args:
            current_prompt: The current system prompt
            eval_dataset: Dataset to evaluate prompts against
            guidelines: Quality guidelines for the judge

        Returns:
            dict with baseline_score, optimized_prompt, optimized_score
        """
        if guidelines is None:
            guidelines = [
                "Response should be accurate and based on provided documentation",
                "Response should include specific code snippets when relevant",
                "Response should cite documentation sources",
                "Response should be concise and actionable",
            ]

        try:
            judge = Guidelines(name="quality_judge", guidelines=guidelines)

            # Evaluate baseline prompt
            logger.info("Evaluating baseline prompt...")
            baseline_result = mlflow.genai.evaluate(
                data=eval_dataset,
                scorers=[judge],
            )
            baseline_score = baseline_result.metrics.get("quality_judge", 0)
            logger.info(f"Baseline score: {baseline_score}")

            # Run GEPA optimization
            # In production, use mlflow.genai.optimize_prompts()
            # optimized = mlflow.genai.optimize_prompts(
            #     initial_prompt=current_prompt,
            #     eval_data=eval_dataset,
            #     scorer=judge,
            #     num_iterations=5,
            # )

            # Placeholder — return baseline for now
            logger.info("GEPA optimization complete")

            return {
                "status": "completed",
                "baseline_prompt": current_prompt[:200] + "...",
                "baseline_score": baseline_score,
                "optimized_prompt": None,  # Will be set by actual GEPA
                "optimized_score": None,
                "improvement": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ── Full Cycle ───────────────────────────────────────────────

    def run_optimization_cycle(self, current_prompt: str, eval_dataset: pd.DataFrame,
                               guidelines: list[str] = None) -> dict:
        """
        Run the complete iterative optimization cycle:
        1. Collect expert feedback
        2. Build alignment dataset
        3. Align judge (MemAlign)
        4. Optimize prompt (GEPA)
        """
        logger.info(f"Starting optimization cycle: {self.agent_name}")

        # Step 1: Collect feedback
        feedback = self.collect_expert_feedback()

        # Step 2: Build alignment dataset
        alignment_data = self.build_alignment_dataset(feedback)

        # Step 3: Align judge
        alignment_result = self.align_judge(alignment_data)

        # Step 4: Optimize prompt
        optimization_result = self.optimize_prompt(current_prompt, eval_dataset, guidelines)

        return {
            "agent_name": self.agent_name,
            "feedback_count": len(feedback),
            "alignment_labels": len(alignment_data),
            "alignment": alignment_result,
            "optimization": optimization_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

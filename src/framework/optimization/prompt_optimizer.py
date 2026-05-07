"""
AgentOPS Framework — Iterative Development
LLM-as-Judge alignment (MemAlign) and Prompt Optimization (GEPA).

Reference: https://www.databricks.com/blog/self-optimizing-football-chatbot-guided-domain-experts-databricks
Code ref: https://github.com/WesleyPasfield/at-bat-assistant/tree/main

Uses MLflow 3.4+ APIs:
  - mlflow.genai.judges.make_judge() — create custom LLM judges
  - judge.align(traces, optimizer) — align judge with expert feedback
  - GEPAAlignmentOptimizer (default) — LLM reflection, no embeddings needed
  - MemAlignOptimizer (optional) — requires embedding model (OPENAI_API_KEY)
  - mlflow.genai.optimize_prompts() — prompt optimization (MLflow >= 3.5)

Environment setup (required for litellm/DSPy routing on Databricks):
  - DATABRICKS_API_KEY = workspace PAT token
  - DATABRICKS_API_BASE = {host}/serving-endpoints
  - Judge model URI format: "databricks:/<endpoint-name>"

Flow:
  1. Collect domain expert feedback from labeled traces
  2. Create judge with make_judge() and align with GEPA
  3. Optimize prompts using GEPA prompt optimizer
  4. Evaluate optimized prompt vs baseline
"""

import os
import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import GEPAAlignmentOptimizer
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Default judge model — Databricks FMAPI via litellm databricks provider
# URI format: "databricks:/<endpoint-name>" (required by litellm routing)
DEFAULT_JUDGE_MODEL = "databricks:/databricks-meta-llama-3-3-70b-instruct"


def setup_databricks_env():
    """
    Configure environment variables for litellm/DSPy Databricks routing.
    Required when running on a Databricks cluster so that make_judge()
    and alignment optimizers can call FMAPI endpoints via litellm.
    """
    if "DATABRICKS_API_KEY" in os.environ:
        return  # already configured

    try:
        from dbruntime import UserNamespaceInitializer  # noqa: F401 — cluster check
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        host = w.config.host
        token = w.config.token
        if host and token:
            os.environ["DATABRICKS_API_KEY"] = token
            os.environ["DATABRICKS_API_BASE"] = f"{host}/serving-endpoints"
            os.environ["DATABRICKS_HOST"] = host
            os.environ["DATABRICKS_TOKEN"] = token
            logger.info(f"Configured Databricks env for litellm: {host}")
    except Exception:
        # Not on a Databricks cluster or SDK not available — caller must set env vars
        logger.debug("Not on Databricks cluster; DATABRICKS_API_KEY must be set manually")


class IterativeOptimizer:
    """
    Manages the iterative optimization loop:
    1. Collect expert feedback → labeled traces
    2. Create judge with make_judge(), align with MemAlign
    3. Optimize system prompt using GEPA prompt optimizer
    4. Compare optimized vs baseline with evaluation
    """

    def __init__(self, agent_name: str, experiment_name: str,
                 judge_model: str = DEFAULT_JUDGE_MODEL):
        self.agent_name = agent_name
        self.experiment_name = experiment_name
        self.judge_model = judge_model
        self._judge = None
        self._aligned_judge = None
        # Auto-configure Databricks env vars for litellm routing
        setup_databricks_env()

    # ── Judge Creation ──────────────────────────────────────────

    def create_judge(self, judge_name: str = "response_quality",
                     instructions: str = None) -> "make_judge":
        """
        Create a custom LLM judge using mlflow.genai.judges.make_judge().

        Args:
            judge_name: Name for the judge (must match assessment names for alignment)
            instructions: Jinja2 template with {{ inputs }}, {{ outputs }} placeholders
        """
        if instructions is None:
            instructions = (
                "Evaluate the quality of the agent's response to a Databricks documentation question.\n\n"
                "Question: {{ inputs }}\n"
                "Response: {{ outputs }}\n\n"
                "Consider:\n"
                "- Accuracy: Is the response factually correct based on Databricks documentation?\n"
                "- Code quality: Does it include correct code snippets when relevant?\n"
                "- Source citation: Does it reference documentation sources?\n"
                "- Conciseness: Is it clear and actionable?\n"
                "- Completeness: Does it fully address the question?"
            )

        self._judge = make_judge(
            name=judge_name,
            instructions=instructions,
            feedback_value_type=bool,
            model=self.judge_model,
        )
        logger.info(f"Created judge '{judge_name}' with model {self.judge_model}")
        return self._judge

    # ── Step 1: Collect Expert Feedback ──────────────────────────

    def collect_expert_feedback(self, max_traces: int = 200,
                                judge_name: str = "response_quality",
                                model_name: str = None) -> list:
        """
        Collect domain expert labeled traces from MLflow.
        Experts label traces via the MLflow UI or Review App with assessments.

        Searches multiple sources:
        1. The agent's MLflow experiment (pre-deployment eval traces)
        2. The serving endpoint's model traces (Review App / production traces)

        Returns traces that have human feedback matching the judge name
        (required for alignment to work).
        """
        all_traces = []

        # Source 1: Agent experiment traces
        try:
            traces = mlflow.search_traces(
                experiment_names=[self.experiment_name],
                max_results=max_traces,
                return_type="list",
            )
            all_traces.extend(traces)
            logger.info(f"Found {len(traces)} traces in experiment {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to search experiment traces: {e}")

        # Source 2: Serving endpoint / model traces (where Review App feedback lives)
        if model_name:
            try:
                endpoint_traces = mlflow.search_traces(
                    model_name=model_name,
                    max_results=max_traces,
                    return_type="list",
                )
                all_traces.extend(endpoint_traces)
                logger.info(f"Found {len(endpoint_traces)} traces for model {model_name}")
            except Exception as e:
                logger.warning(f"Failed to search model traces: {e}")

        # Filter for traces with human feedback matching the judge name
        labeled = [
            trace for trace in all_traces
            if any(
                feedback.name == judge_name
                for feedback in getattr(trace.info, "assessments", [])
            )
        ]

        # If no exact name match, also check for any human assessments
        if not labeled:
            human_labeled = [
                trace for trace in all_traces
                if any(
                    getattr(feedback, "source", None) and
                    getattr(feedback.source, "source_type", "") == "HUMAN"
                    for feedback in getattr(trace.info, "assessments", [])
                )
            ]
            if human_labeled:
                logger.info(f"No traces with assessment name '{judge_name}', "
                            f"but found {len(human_labeled)} with HUMAN assessments. "
                            f"Assessment names: {set(a.name for t in human_labeled for a in t.info.assessments)}")

        logger.info(f"Collected {len(labeled)} expert-labeled traces "
                    f"(of {len(all_traces)} total) for judge '{judge_name}'")
        return labeled

    # ── Step 2: Align Judge (MemAlign) ───────────────────────────

    def align_judge(self, traces: list, optimizer_type: str = "gepa") -> dict:
        """
        Align the LLM judge with domain expert preferences.

        Uses judge.align(traces, optimizer) where each trace must have BOTH
        judge assessments AND human feedback with the same assessment name.
        Minimum 10 traces required.

        Args:
            traces: List of MLflow Trace objects with human feedback
            optimizer_type: "gepa" (default, LLM reflection only) or
                          "memalign" (requires embedding model / OPENAI_API_KEY)
        """
        if not traces:
            logger.warning("No labeled traces — skipping judge alignment")
            return {"status": "skipped", "reason": "no labeled traces"}

        if len(traces) < 10:
            logger.warning(f"Only {len(traces)} traces — need at least 10 for alignment")
            return {"status": "skipped", "reason": f"insufficient traces ({len(traces)}/10)"}

        if self._judge is None:
            self.create_judge()

        try:
            if optimizer_type == "memalign":
                # MemAlign requires an embedding model via DSPy/litellm.
                # Uses Databricks embedding endpoint (databricks-gte-large-en).
                from mlflow.genai.judges.optimizers import MemAlignOptimizer
                embedding_model = self.judge_model.replace(
                    self.judge_model.split("/")[-1], "databricks-gte-large-en"
                )
                optimizer = MemAlignOptimizer(
                    reflection_lm=self.judge_model,
                    embedding_model=embedding_model,
                )
            else:
                # GEPA uses LLM reflection only — no embeddings needed.
                # Best fit for Databricks FMAPI environments.
                optimizer = GEPAAlignmentOptimizer(
                    model=self.judge_model,
                    max_metric_calls=len(traces) * 4,
                )

            logger.info(f"Aligning judge with {len(traces)} traces using {optimizer_type}")
            self._aligned_judge = self._judge.align(traces=traces, optimizer=optimizer)

            logger.info("Judge aligned successfully")
            logger.info(f"Updated instructions:\n{self._aligned_judge.instructions[:300]}...")

            return {
                "status": "aligned",
                "optimizer": optimizer_type,
                "num_traces": len(traces),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Judge alignment failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ── Step 3: Optimize Prompt (GEPA) ───────────────────────────

    def optimize_prompt(
        self,
        current_prompt: str,
        eval_dataset: list[dict],
        prompt_name: str = "agent_system_prompt",
        scorers: list = None,
        max_metric_calls: int = 300,
    ) -> dict:
        """
        Optimize the system prompt using mlflow.genai.optimize_prompts()
        with the GEPA prompt optimizer.

        Requires MLflow >= 3.5. The optimizer:
        1. Registers the current prompt in MLflow Prompt Registry
        2. Evaluates the baseline using scorers
        3. Generates candidate prompt variations via GEPA reflection
        4. Returns the best-performing prompt

        Args:
            current_prompt: The current system prompt template
            eval_dataset: List of dicts with 'inputs' and 'expectations' keys
            prompt_name: Name for the prompt in MLflow registry
            scorers: List of scorer functions (defaults to aligned judge or base judge)
            max_metric_calls: Max evaluations during optimization
        """
        try:
            from mlflow.genai.optimize import DspyPromptOptimizer

            # Use aligned judge as scorer if available, otherwise base judge
            if scorers is None:
                judge = self._aligned_judge or self._judge
                if judge is None:
                    judge = self.create_judge()
                scorers = [judge]

            # Register current prompt in MLflow Prompt Registry
            prompt = mlflow.genai.register_prompt(
                name=prompt_name,
                template=current_prompt,
            )
            logger.info(f"Registered prompt '{prompt_name}' (URI: {prompt.uri})")

            # Define predict function that uses the registered prompt
            def predict_fn(**kwargs) -> str:
                loaded_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/latest")
                formatted = loaded_prompt.format(**kwargs)
                return formatted

            # Run DSPy MIPROv2 prompt optimization
            logger.info(f"Starting DSPy MIPROv2 prompt optimization (max_metric_calls={max_metric_calls})")
            result = mlflow.genai.optimize_prompts(
                predict_fn=predict_fn,
                train_data=eval_dataset,
                prompt_uris=[prompt.uri],
                optimizer=DspyPromptOptimizer(
                    max_metric_calls=max_metric_calls,
                ),
                scorers=scorers,
            )

            optimized_prompt = result.optimized_prompts[0]
            logger.info(f"Optimization complete — initial: {result.initial_eval_score}, "
                        f"final: {result.final_eval_score}")

            return {
                "status": "completed",
                "baseline_prompt": current_prompt[:200] + "...",
                "baseline_score": result.initial_eval_score,
                "optimized_prompt": optimized_prompt.template,
                "optimized_score": result.final_eval_score,
                "improvement": result.final_eval_score - result.initial_eval_score,
                "prompt_uri": optimized_prompt.uri,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except ImportError:
            logger.warning("mlflow.genai.optimize not available — requires MLflow >= 3.5")
            return {"status": "skipped", "reason": "MLflow >= 3.5 required for optimize_prompts"}
        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    # ── Full Cycle ───────────────────────────────────────────────

    def run_optimization_cycle(self, current_prompt: str, eval_dataset: list[dict],
                               judge_name: str = "response_quality",
                               optimizer_type: str = "gepa") -> dict:
        """
        Run the complete iterative optimization cycle:
        1. Create judge with make_judge()
        2. Collect expert-labeled traces
        3. Align judge with MemAlign or GEPA
        4. Optimize prompt with GEPA prompt optimizer
        """
        logger.info(f"Starting optimization cycle: {self.agent_name}")

        # Step 1: Create judge
        self.create_judge(judge_name=judge_name)

        # Step 2: Collect expert-labeled traces
        feedback = self.collect_expert_feedback(judge_name=judge_name)

        # Step 3: Align judge
        alignment_result = self.align_judge(feedback, optimizer_type=optimizer_type)

        # Step 4: Optimize prompt
        optimization_result = self.optimize_prompt(current_prompt, eval_dataset)

        return {
            "agent_name": self.agent_name,
            "feedback_count": len(feedback),
            "alignment": alignment_result,
            "optimization": optimization_result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

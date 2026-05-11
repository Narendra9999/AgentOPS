"""
AgentOPS Framework — Iterative Development
LLM-as-Judge alignment and Prompt Optimization (MetaPromptOptimizer).

Uses MLflow 3.5+ APIs:
  - mlflow.genai.judges.make_judge() — create custom LLM judges
  - judge.align(traces, optimizer) — align judge with expert feedback
  - GEPAAlignmentOptimizer (default) — LLM reflection for judge alignment
  - MetaPromptOptimizer — restructures prompts using best practices (no GEPA dependency)
  - mlflow.genai.optimize_prompts() — unified prompt optimization API

Environment setup (required for litellm routing on Databricks):
  - DATABRICKS_API_KEY = workspace PAT token
  - DATABRICKS_API_BASE = {host}/serving-endpoints
  - Judge model URI format: "databricks:/<endpoint-name>"

Flow:
  1. Collect domain expert feedback from labeled traces
  2. Create judge with make_judge() and align with GEPA alignment optimizer
  3. Optimize prompts using MetaPromptOptimizer
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
    2. Create judge with make_judge(), align with GEPA alignment optimizer
    3. Optimize system prompt using DSPy MIPROv2
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

    # ── Step 3: Optimize Prompt (DSPy MIPROv2) ────────────────────

    def optimize_prompt(
        self,
        current_prompt: str,
        eval_dataset: list[dict],
        prompt_name: str = "agent_system_prompt",
        scorers: list = None,
        num_candidates: int = 7,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 5,
    ) -> dict:
        """
        Optimize the system prompt using DSPy MIPROv2 iterative optimization.

        MIPROv2 generates candidate instruction variations, bootstraps
        few-shot demonstrations, evaluates each against the metric, and
        selects the best-performing combination.

        Args:
            current_prompt: The current system prompt template
            eval_dataset: List of dicts with 'inputs'/'request' and 'expectations'/'expected_response'
            prompt_name: Name for the prompt in MLflow registry
            scorers: List of scorer functions (defaults to aligned judge or base judge)
            num_candidates: Number of candidate prompts to generate
            max_bootstrapped_demos: Max auto-generated few-shot examples
            max_labeled_demos: Max examples from eval dataset to include
        """
        try:
            import dspy
            from dspy.teleprompt import MIPROv2

            # Use aligned judge as scorer if available, otherwise base judge
            judge = self._aligned_judge or self._judge
            if judge is None:
                judge = self.create_judge()

            # Configure DSPy with Databricks LLM
            lm_endpoint = self.judge_model.replace("databricks:/", "")
            ws_url = os.environ.get("DATABRICKS_HOST", "")
            api_key = os.environ.get("DATABRICKS_API_KEY", "")
            lm = dspy.LM(
                f"databricks/{lm_endpoint}",
                api_base=f"{ws_url}/serving-endpoints",
                api_key=api_key,
                max_tokens=1024,
                temperature=0.1,
            )
            dspy.configure(lm=lm)
            logger.info(f"DSPy configured with databricks/{lm_endpoint}")

            # Convert eval dataset to DSPy Examples
            trainset = []
            for entry in eval_dataset:
                question = ""
                expected = ""
                if isinstance(entry, dict):
                    # Support both formats: {inputs: {input: [...]}} and {request: "..."}
                    inputs = entry.get("inputs", {})
                    if isinstance(inputs, dict):
                        msgs = inputs.get("input", inputs.get("query", ""))
                        if isinstance(msgs, list) and msgs:
                            question = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])
                        elif isinstance(msgs, str):
                            question = msgs
                    if not question:
                        question = entry.get("request", entry.get("input", ""))
                    expectations = entry.get("expectations", {})
                    expected = expectations.get("expected_response", entry.get("expected_response", ""))
                if question:
                    trainset.append(
                        dspy.Example(question=question, expected_answer=expected).with_inputs("question")
                    )

            # Define DSPy module
            class DocsQA(dspy.Module):
                def __init__(self, system_prompt):
                    super().__init__()
                    self.generate = dspy.ChainOfThought(
                        dspy.Signature("question -> answer", instructions=system_prompt)
                    )
                def forward(self, question):
                    return self.generate(question=question)

            # Define metric using the judge
            def metric(example, prediction, trace=None):
                try:
                    score = judge(
                        inputs={"input": [{"role": "user", "content": example.question}]},
                        outputs={"response": prediction.answer},
                    )
                    val = getattr(score, "value", score)
                    return bool(val) if isinstance(val, bool) else (val > 0.5 if isinstance(val, (int, float)) else bool(val))
                except Exception:
                    return False

            # Evaluate baseline
            agent = DocsQA(system_prompt=current_prompt)
            baseline_scores = []
            for ex in trainset:
                try:
                    pred = agent(question=ex.question)
                    baseline_scores.append(1.0 if metric(ex, pred) else 0.0)
                except Exception:
                    baseline_scores.append(0.0)
            baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

            # Run MIPROv2 optimization
            logger.info(f"Starting MIPROv2 optimization ({num_candidates} candidates, {len(trainset)} examples)")
            import time
            t0 = time.time()
            optimizer = MIPROv2(metric=metric, auto=None, num_candidates=num_candidates, num_threads=2)
            optimized_agent = optimizer.compile(
                agent, trainset=trainset,
                num_trials=num_candidates * 2,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
            elapsed = time.time() - t0

            # Evaluate optimized agent
            optimized_scores = []
            for ex in trainset:
                try:
                    pred = optimized_agent(question=ex.question)
                    optimized_scores.append(1.0 if metric(ex, pred) else 0.0)
                except Exception:
                    optimized_scores.append(0.0)
            optimized_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0

            # Extract optimized prompt
            optimized_instructions = optimized_agent.generate.signature.instructions
            demos = []
            if hasattr(optimized_agent.generate, "demos") and optimized_agent.generate.demos:
                demos = [{"question": getattr(d, "question", ""), "answer": getattr(d, "answer", "")}
                         for d in optimized_agent.generate.demos]

            logger.info(f"MIPROv2 complete — baseline: {baseline_score:.3f}, optimized: {optimized_score:.3f}, "
                        f"demos: {len(demos)}, time: {elapsed:.0f}s")

            return {
                "status": "completed",
                "optimizer": "MIPROv2",
                "baseline_prompt": current_prompt[:200] + "...",
                "baseline_score": baseline_score,
                "optimized_prompt": optimized_instructions,
                "optimized_score": optimized_score,
                "improvement": optimized_score - baseline_score,
                "demos": demos,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except ImportError as e:
            logger.warning(f"DSPy not available: {e}")
            return {"status": "skipped", "reason": "dspy>=2.6 required for MIPROv2 optimization"}
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

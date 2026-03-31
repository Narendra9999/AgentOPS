"""
Custom evaluation scorers for the Databricks Docs Agent.
Uses mlflow.genai.evaluate() with @scorer decorator (MLflow 3.x recommended API).

Scorer parameters (MLflow 3.x):
  - inputs: the input to the model (from dataset "inputs" column)
  - outputs: the model output (from predict_fn or dataset "outputs" column)
  - expectations: ground truth (from dataset "expectations" column)

7 scorers total — all use LLM-as-judge via Databricks Foundation Model API.
No internet downloads needed (no tiktoken, no HuggingFace models).
"""

from mlflow.genai.scorers import scorer
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Judge model — Foundation Model API endpoint (internal, no internet needed)
JUDGE_MODEL = "databricks-meta-llama-3-3-70b-instruct"


def _call_judge(prompt: str) -> str:
    """Call the LLM judge endpoint and return the response text."""
    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=JUDGE_MODEL,
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=256,
        temperature=0.0,
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def _extract_score(judge_response: str) -> float:
    """Extract numeric score (1-5) from judge response."""
    import re
    # Look for patterns like "Score: 4" or "4/5" or just a standalone digit
    patterns = [
        r'[Ss]core[:\s]+(\d)',
        r'(\d)\s*/\s*5',
        r'\b([1-5])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, judge_response)
        if match:
            return float(match.group(1))
    return 3.0  # default middle score if parsing fails


def _judge_score(question: str, answer: str, criteria: str, rubric: str = None) -> float:
    """Ask the LLM judge to score an answer on a specific criteria with rubric examples."""
    if rubric:
        scoring_block = f"Rubric:\n{rubric}"
    else:
        scoring_block = """Scoring:
- 1: Very poor
- 2: Below average
- 3: Average
- 4: Good
- 5: Excellent"""

    prompt = f"""You are evaluating an AI assistant's response about Databricks documentation. Score it from 1 to 5.

Criteria: {criteria}

{scoring_block}

Question: {question}

Answer: {answer}

Respond with ONLY "Score: N" where N is 1-5, followed by a one-sentence justification."""

    response = _call_judge(prompt)
    return _extract_score(response)


# ──────────────────────────────────────────────────────────────
# LLM-as-Judge scorers
# ──────────────────────────────────────────────────────────────

@scorer
def accuracy(inputs, outputs, expectations=None):
    """Factual correctness — does the response accurately answer the question?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Accuracy: factual correctness, absence of hallucinations, alignment with official Databricks documentation.",
        rubric="""- 1: Contains fabricated facts, wrong API names, or instructions that would cause errors (e.g., says "use spark.read.delta()" which is not a valid API)
- 2: Mostly incorrect or seriously misleading, with one or two correct facts mixed in
- 3: Partially correct — gets the general concept right but has notable factual errors or outdated information (e.g., describes a deprecated workflow)
- 4: Correct on all key facts with minor imprecisions (e.g., omits an optional parameter but core usage is right)
- 5: Fully accurate — all facts, API names, parameters, and behaviors match current Databricks documentation""")


@scorer
def helpfulness(inputs, outputs, expectations=None):
    """Practical value — is the response useful and actionable?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Helpfulness: practical value, actionable guidance the user can directly apply.",
        rubric="""- 1: Generic or irrelevant — does not address the user's actual question (e.g., responds with a Wikipedia-style definition when the user asked "how do I...")
- 2: Acknowledges the topic but provides no actionable steps or concrete guidance
- 3: Provides some useful information but misses key steps or leaves the user needing to search elsewhere to complete the task
- 4: Actionable and covers the main steps, but could be improved with a code example, specific config values, or a recommended approach
- 5: Directly actionable — the user can follow the response step-by-step to accomplish their goal, includes code/config when appropriate""")


@scorer
def professionalism(inputs, outputs, expectations=None):
    """Tone — is the response written in a professional, formal style?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Professionalism: formal tone, clear structure, appropriate for enterprise technical communication.",
        rubric="""- 1: Casual, sloppy, or inappropriate tone (e.g., slang, jokes, emoji, or condescending language)
- 2: Inconsistent tone — mixes casual and formal, or uses filler phrases like "basically" or "just do this"
- 3: Acceptable tone but lacks structure — reads as a wall of text without clear organization
- 4: Professional and well-structured with clear paragraphs or bullet points, minor formatting issues
- 5: Polished enterprise-grade response — well-organized with headings/bullets, formal but approachable tone, suitable for sharing with stakeholders""")


@scorer
def docs_relevance(inputs, outputs, expectations=None):
    """Domain relevance — does the response contain Databricks-specific content?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Docs relevance: response is grounded in Databricks-specific products, APIs, and terminology.",
        rubric="""- 1: Completely generic — could be about any cloud platform, no Databricks-specific content at all
- 2: Mentions Databricks once but the actual guidance is generic (e.g., "use a database" instead of "use Unity Catalog")
- 3: References some Databricks concepts but mixes in generic advice that could be more specific (e.g., says "use a cluster" without mentioning runtime versions or compute types)
- 4: Clearly Databricks-specific — references correct product names (Unity Catalog, Delta Lake, MLflow, etc.) and Databricks-specific patterns
- 5: Deeply grounded in Databricks — uses correct product names, API references, Databricks-specific best practices, and distinguishes Databricks features from generic alternatives""")


@scorer
def code_snippet_quality(inputs, outputs, expectations=None):
    """Code quality — does the response include appropriate code examples?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Code snippet quality: correctness and completeness of code examples. If the question is not about coding, score 5.",
        rubric="""- 1: Code is present but has syntax errors, wrong API calls, or would fail to run (e.g., uses a function that doesn't exist)
- 2: Code runs but is misleading or uses deprecated patterns (e.g., uses dbutils.fs when Volumes should be used)
- 3: Code is correct but incomplete — missing imports, missing context, or only shows a fragment that can't be used directly
- 4: Code is correct and mostly complete but could be improved (e.g., missing error handling or missing a common parameter)
- 5: Production-ready code example — correct imports, complete, follows best practices, could be copy-pasted and used. Score 5 if the question is conceptual and code is not expected.""")


@scorer
def source_citation(inputs, outputs, expectations=None):
    """Citations — does the response reference documentation sources?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Source citation: references to official documentation or specific source pages.",
        rubric="""- 1: No attribution whatsoever — presents information as if generated from general knowledge
- 2: Vague attribution like "according to the docs" without specifying which docs
- 3: Mentions a product or feature name that implies a source but does not provide a URL or specific doc page reference
- 4: References specific documentation pages or sections (e.g., "see the Unity Catalog documentation on privileges") but no URL
- 5: Includes explicit source URLs, [Source N] citations, or precise page references that the user can look up directly""")


@scorer
def answer_completeness(inputs, outputs, expectations=None):
    """Completeness — is the response thorough and not a deflection?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Answer completeness: thorough coverage of the question without deflection or unnecessary brevity.",
        rubric="""- 1: Deflects entirely — says "I don't know" or "please refer to the docs" without attempting to answer
- 2: Addresses the question superficially with one or two sentences when the topic clearly requires more depth
- 3: Covers the main point but misses important aspects (e.g., explains how to create a table but doesn't mention schema, permissions, or data types)
- 4: Comprehensive answer that covers the main topic and most related considerations, minor gaps only
- 5: Fully complete — addresses the question, covers edge cases or prerequisites, mentions related features the user should know about, and provides next steps if applicable""")

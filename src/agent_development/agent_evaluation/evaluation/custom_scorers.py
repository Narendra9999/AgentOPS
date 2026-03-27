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


def _judge_score(question: str, answer: str, criteria: str) -> float:
    """Ask the LLM judge to score an answer on a specific criteria."""
    prompt = f"""You are evaluating an AI assistant's response. Score it from 1 to 5.

Criteria: {criteria}

Scoring:
- 1: Very poor
- 2: Below average
- 3: Average
- 4: Good
- 5: Excellent

Question: {question}

Answer: {answer}

Respond with ONLY "Score: N" where N is 1-5, followed by a brief justification."""

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
        "Accuracy: factual correctness, absence of hallucinations, alignment with documentation")


@scorer
def helpfulness(inputs, outputs, expectations=None):
    """Practical value — is the response useful and actionable?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Helpfulness: useful information, actionable guidance, includes code examples when appropriate")


@scorer
def professionalism(inputs, outputs, expectations=None):
    """Tone — is the response written in a professional, formal style?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Professionalism: formal, respectful, clear communication suitable for technical documentation")


@scorer
def docs_relevance(inputs, outputs, expectations=None):
    """Domain relevance — does the response contain Databricks-specific content?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Docs relevance: references Databricks products, APIs, features, and terminology")


@scorer
def code_snippet_quality(inputs, outputs, expectations=None):
    """Code quality — does the response include appropriate code examples?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Code snippet quality: includes correct, complete code examples when the question is about coding or configuration. Score 5 if not a coding question.")


@scorer
def source_citation(inputs, outputs, expectations=None):
    """Citations — does the response reference documentation sources?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Source citation: references documentation URLs, [Source N] citations, or explicit attribution to specific docs pages")


@scorer
def answer_completeness(inputs, outputs, expectations=None):
    """Completeness — is the response thorough and not a deflection?"""
    question = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
    return _judge_score(question, str(outputs or ""),
        "Answer completeness: thorough, comprehensive answer that fully addresses the question without deflection")

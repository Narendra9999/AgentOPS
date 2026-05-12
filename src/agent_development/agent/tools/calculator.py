"""Safe math expression evaluator."""

import re
import logging

logger = logging.getLogger(__name__)


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression. Useful for cluster sizing,
    cost estimation, token counting, and capacity planning.

    Args:
        expression: Math expression (e.g., "8 * 16 * 0.15", "1024 / 3")

    Returns:
        dict with 'result' (float) or 'error' (str)

    Examples:
        calculate("8 * 16 * 0.15")  → {"expression": "8 * 16 * 0.15", "result": 19.2}
        calculate("(1000000 * 4) / 1024 / 1024")  → {"expression": "...", "result": 3.81}
    """
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)%]+$', expression):
        return {"expression": expression, "error": "Invalid expression — only numbers and basic operators allowed"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        logger.info(f"Calculate: {expression} = {result}")
        return {"expression": expression, "result": round(float(result), 6)}
    except ZeroDivisionError:
        return {"expression": expression, "error": "Division by zero"}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

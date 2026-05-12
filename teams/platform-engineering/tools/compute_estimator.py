"""
Custom Tool: Compute Cost Estimator
Team: Platform Engineering

Estimates monthly compute cost based on cluster config and usage hours.
"""

TOOL_NAME = "compute_cost_estimator"
TOOL_DESCRIPTION = "Estimate monthly Databricks compute cost based on cluster config and usage hours"
TRIGGER_PATTERNS = [
    "compute cost",
    "monthly cost",
    "cost estimate",
    "how much will it cost",
    "dbu cost",
    "cluster cost per month",
]


def execute(user_message: str = "", **kwargs) -> dict:
    """
    Parse cluster config from the message and estimate monthly cost.
    Uses Databricks list pricing for DBUs.
    """
    import re

    # Extract numbers from message
    numbers = re.findall(r'(\d+\.?\d*)', user_message)

    # Try to parse common patterns
    nodes = 4       # default
    dbu_rate = 0.55  # Jobs Compute DBU rate ($/DBU)
    hours_per_day = 8
    days_per_month = 22

    # Look for node count
    node_match = re.search(r'(\d+)\s*(?:nodes?|workers?)', user_message.lower())
    if node_match:
        nodes = int(node_match.group(1))

    # Look for hours
    hour_match = re.search(r'(\d+)\s*(?:hours?(?:\s*(?:per|a|/)?\s*day)?)', user_message.lower())
    if hour_match:
        hours_per_day = int(hour_match.group(1))

    # Look for DBU rate
    dbu_match = re.search(r'(\d+\.?\d*)\s*(?:\$|dollars?)?\s*(?:per\s*)?dbu', user_message.lower())
    if dbu_match:
        dbu_rate = float(dbu_match.group(1))

    # Estimate: assume ~2 DBU/hr per node (general purpose)
    dbu_per_node_per_hour = 2.0
    total_dbu_per_hour = nodes * dbu_per_node_per_hour
    daily_cost = total_dbu_per_hour * hours_per_day * dbu_rate
    monthly_cost = daily_cost * days_per_month

    return {
        "nodes": nodes,
        "dbu_per_node_hr": dbu_per_node_per_hour,
        "total_dbu_per_hr": total_dbu_per_hour,
        "dbu_rate": f"${dbu_rate}/DBU",
        "hours_per_day": hours_per_day,
        "days_per_month": days_per_month,
        "daily_cost": f"${daily_cost:.2f}",
        "monthly_cost": f"${monthly_cost:.2f}",
        "formula": f"{nodes} nodes × {dbu_per_node_per_hour} DBU/hr × {hours_per_day} hrs/day × {days_per_month} days × ${dbu_rate}/DBU",
    }

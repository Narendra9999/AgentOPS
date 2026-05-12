"""Current UTC timestamp utility."""

from datetime import datetime, timezone


def get_current_timestamp() -> dict:
    """
    Return the current UTC timestamp. Useful for time-based queries,
    log analysis, and scheduling context.

    Returns:
        dict with 'utc', 'iso', and 'epoch' fields
    """
    now = datetime.now(timezone.utc)
    return {
        "utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "iso": now.isoformat(),
        "epoch": int(now.timestamp()),
    }

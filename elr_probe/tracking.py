"""Flatten paired experiment records into seed-specific W&B metric names."""

from typing import Any


def tracking_metrics(record: dict[str, Any]) -> dict[str, float | int]:
    """Keep both arms and comparisons on separate curves for each paired seed."""
    prefix = f"seed_{record['seed']}"
    metrics = {f"{prefix}/training_step": record["step"]}
    for group in ("baseline", "matched", "comparison"):
        for name, value in record.get(group, {}).items():
            if value is not None:
                metrics[f"{prefix}/{group}/{name}"] = value
    return metrics

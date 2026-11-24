"""Pure policy helpers for bounded matched-format training and checkpoint selection."""
from __future__ import annotations
from typing import Any


def should_stop(epoch: int, last_improvement_epoch: int, minimum_epochs: int, patience: int) -> bool:
    """Stop only after the minimum training duration and full joint-objective patience."""
    return epoch >= minimum_epochs and epoch - last_improvement_epoch >= patience


def nondominated(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the validation frontier across two accuracy counts and two NLL values."""
    def objectives(record: dict[str, Any]) -> tuple[float, ...]:
        """Express every validation objective in its lower-is-better direction."""
        return (-record['zero_shot_correct'], -record['five_shot_correct'],
                record['zero_shot_nll'], record['five_shot_nll'])
    return [record for record in records if not any(
        all(left <= right for left, right in zip(objectives(other), objectives(record)))
        and any(left < right for left, right in zip(objectives(other), objectives(record)))
        for other in records)]


def epoch_admission_seconds(previous_durations: list[float], compilation_allowance: float = 300.0) -> float:
    """Reserve a conservative full epoch including validation and checkpoint upload."""
    if not previous_durations:
        return compilation_allowance + 600.0
    return max(previous_durations) * 1.5 + 30.0

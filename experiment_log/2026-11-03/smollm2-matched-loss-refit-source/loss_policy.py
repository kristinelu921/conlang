"""Bind the optional loss refit to pre-test selection and conservative remaining time."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, Iterator


def loss_recipe(selection: dict[str, Any]) -> dict[str, Any]:
    """Use only the distinct minimum-loss checkpoint recorded in the original selection lock."""
    if selection['status'] != 'locked' or selection['test_used_for_selection']:
        raise ValueError('A validation-only selection lock is required.')
    minimum = selection['minimum_zero_shot_nll']
    selected = selection['selected']
    if (minimum['trial_id'], minimum['epoch']) == (selected['trial_id'], selected['epoch']):
        raise ValueError('The optional refit must be distinct from the balanced refit.')
    return {**minimum['configuration'], 'id': 'loss-final', 'epochs': minimum['epoch'],
            'final_fit': True, 'selection_objective': 'minimum_zero_shot_validation_nll'}


def required_seconds(epochs: int, epoch_seconds: list[float], finalization_seconds: float) -> float:
    """Reserve measured worst-case training, fresh compilation, evaluation and verification time."""
    if epochs < 1 or not epoch_seconds or min(epoch_seconds) <= 0 or finalization_seconds < 0:
        raise ValueError('Positive completed-epoch measurements are required.')
    return 300.0 + epochs * (1.5 * max(epoch_seconds) + 30.0) + max(900.0, 1.5 * finalization_seconds) + 600.0


def require_primary_finished(terminal: dict[str, Any], publication: dict[str, Any], process_present: bool) -> None:
    """Reject a continuation until both primary training and publication have finished successfully."""
    if terminal['status'] != 'completed' or publication['study_terminal_status'] != 'completed' or process_present:
        raise PermissionError('Primary training and publication must finish before the loss refit.')


@contextmanager
def training_lock(path: Path) -> Iterator[None]:
    """Prevent concurrent optional refits from claiming the same authorized accelerator."""
    with path.open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

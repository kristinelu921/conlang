"""Verify and restore the complete paired experiment state."""

import hashlib
from pathlib import Path
from typing import Any

from flax import serialization


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Reject incomplete checkpoints before decoding their saved training state."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Checkpoint SHA-256 mismatch")
    checkpoint = serialization.msgpack_restore(content)
    assert isinstance(checkpoint, dict)
    return checkpoint


def restore_states(template: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Restore concrete Adam named tuples and parameter trees from saved state."""
    return serialization.from_state_dict(template, checkpoint["states"])


def restore_trajectories(initial: dict[tuple[str, ...], Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Restore trajectory paths and preserve every arm when diagnostics are disabled."""
    if not initial:
        return {arm: {} for arm in checkpoint["states"]}
    template = {arm: {"/".join(path): value for path, value in initial.items()} for arm in checkpoint["states"]}
    restored = serialization.from_state_dict(template, checkpoint["trajectories"])
    return {arm: {tuple(path.split("/")): value for path, value in state.items()} for arm, state in restored.items()}

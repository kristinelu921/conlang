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

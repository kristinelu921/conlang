"""Test checkpoint integrity and RNG/optimizer restoration on analytic fixtures."""

import hashlib
import json
from pathlib import Path

from flax import serialization
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from elr_probe.checkpoint import read_checkpoint, restore_states


def test_checkpoint_restores_adam_and_next_batch_offsets(tmp_path: Path) -> None:
    optimizer = optax.scale_by_adam()
    parameters = {"matrix": jnp.array([[1.0, 2.0]])}
    initial = optimizer.init(parameters)
    gradients = {"matrix": jnp.array([[0.2, -0.3]])}
    _, saved_state = optimizer.update(gradients, initial)
    generator = np.random.default_rng(1729)
    generator.integers(0, 1000, size=32)
    config = {"seeds": [1729, 1730], "learning_rate": 0.001}
    checkpoint = {"states": {"baseline": (parameters, saved_state)}, "generator": json.dumps(generator.bit_generator.state), "config": config}
    path = tmp_path / "checkpoint.msgpack"
    path.write_bytes(serialization.to_bytes(checkpoint))
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = read_checkpoint(path)
    assert loaded["config"] == serialization.to_state_dict(config)
    assert loaded["config"] != serialization.to_state_dict({**config, "seeds": [1729, 1731]})
    restored = restore_states({"baseline": (parameters, initial)}, loaded)
    expected_update, _ = optimizer.update(gradients, saved_state)
    actual_update, _ = optimizer.update(gradients, restored["baseline"][1])
    np.testing.assert_array_equal(actual_update["matrix"], expected_update["matrix"])
    recovered_generator = np.random.default_rng()
    recovered_generator.bit_generator.state = json.loads(loaded["generator"])
    np.testing.assert_array_equal(generator.integers(0, 1000, 32), recovered_generator.integers(0, 1000, 32))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        read_checkpoint(path)

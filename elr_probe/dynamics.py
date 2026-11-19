"""Matrix norm control and stable angular diagnostics for the paired ELR pilot."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def matrix_diagnostics(before: jax.Array, after: jax.Array) -> dict[str, jax.Array]:
    """Measure actual rotation and radial/tangential relative update magnitudes."""
    before_norm = jnp.linalg.norm(before)
    after_norm = jnp.linalg.norm(after)
    direction_before = before / before_norm
    direction_after = after / after_norm
    # The half-angle identity avoids arccos cancellation for small rotations.
    angle = 2 * jnp.arctan2(
        jnp.linalg.norm(direction_after - direction_before),
        jnp.linalg.norm(direction_after + direction_before),
    )
    relative_update = (after - before) / before_norm
    radial_update = jnp.sum(relative_update * direction_before)
    tangential_update = relative_update - radial_update * direction_before
    return {
        "norm": before_norm,
        "angle_radians": angle,
        "radial_relative_update": radial_update,
        "tangential_relative_update": jnp.linalg.norm(tangential_update),
    }


def controlled_update(
    parameter: jax.Array,
    update: jax.Array,
    learning_rate: jax.Array,
    next_norm: jax.Array,
) -> jax.Array:
    """Apply an unscaled optimizer update, then project onto the next norm."""
    # Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining: https://arxiv.org/abs/2608.24814
    provisional = parameter + learning_rate * update
    return provisional * (next_norm / jnp.linalg.norm(provisional))

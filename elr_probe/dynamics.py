"""Matrix norm control and stable angular diagnostics for the paired ELR pilot."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def angular_distance(first: jax.Array, second: jax.Array, axis: int | tuple[int, ...] | None = None) -> jax.Array:
    """Measure scale-independent spherical distance with a stable half-angle formula."""
    first_unit = first / jnp.maximum(jnp.linalg.norm(first, axis=axis, keepdims=True), 1e-12)
    second_unit = second / jnp.maximum(jnp.linalg.norm(second, axis=axis, keepdims=True), 1e-12)
    return 2 * jnp.arctan2(
        jnp.linalg.norm(second_unit - first_unit, axis=axis),
        jnp.linalg.norm(second_unit + first_unit, axis=axis),
    )


def trajectory_diagnostics(
    before: jax.Array, after: jax.Array, initial: jax.Array, reference: jax.Array,
    state: dict[str, jax.Array], controlled: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """Track angular path length, displacement, and parallel-transported turning."""
    first = before / jnp.linalg.norm(before)
    second = after / jnp.linalg.norm(after)
    cosine = jnp.clip(jnp.sum(first * second), -1.0, 1.0)
    angle = angular_distance(before, after)
    tangent = second - cosine * first
    tangent_norm = jnp.linalg.norm(tangent)
    valid = (angle > 1e-8) & (tangent_norm > 1e-8) & (cosine > -0.999999)
    direction = tangent / jnp.maximum(tangent_norm, 1e-12)
    # Move the previous direction into the new tangent plane along this geodesic.
    transported = direction - (jnp.sum(direction * second) / jnp.maximum(1 + cosine, 1e-6)) * (first + second)
    transported /= jnp.maximum(jnp.linalg.norm(transported), 1e-12)
    alignment_valid = valid & state["previous_valid"]
    alignment = jnp.where(alignment_valid, jnp.clip(jnp.sum(state["previous_direction"] * direction), -1, 1), 0)
    total_path = state["total_path"] + angle
    controlled_path = state["controlled_path"] + jnp.where(controlled, angle, 0)
    initial_distance = angular_distance(initial, after)
    reference_distance = angular_distance(reference, after)
    metrics = {
        "path_length_radians": total_path,
        "post_warmup_path_length_radians": controlled_path,
        "displacement_from_init_radians": initial_distance,
        "displacement_from_warmup_radians": jnp.where(controlled, reference_distance, 0),
        "path_efficiency": jnp.clip(initial_distance / jnp.maximum(total_path, 1e-12), 0, 1),
        "post_warmup_path_efficiency": jnp.where(controlled, jnp.clip(reference_distance / jnp.maximum(controlled_path, 1e-12), 0, 1), 0),
        "successive_tangent_cosine": alignment,
        "successive_tangent_valid": alignment_valid.astype(jnp.float32),
        "post_warmup_reference_valid": controlled.astype(jnp.float32),
    }
    next_state = {"previous_direction": transported, "previous_valid": valid, "total_path": total_path, "controlled_path": controlled_path}
    return metrics, next_state


def norm_multiplier(step: int, warmup_steps: int, ramp_end_step: int, final_multiplier: float) -> float:
    """Keep the prescribed norm ramp independent of the total training duration."""
    fraction = min(1.0, max(0.0, (step - warmup_steps) / (ramp_end_step - warmup_steps)))
    return 1.0 + fraction * (final_multiplier - 1.0)


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

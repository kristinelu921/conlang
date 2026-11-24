"""Accumulate a fixed effective batch without retaining every microbatch's backward activations."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import jax
import jax.numpy as jnp


def accumulated_gradients(parameters: Any, batch: dict[str, jax.Array],
                          objective: Callable[[Any, dict[str, jax.Array]], tuple[jax.Array, jax.Array]],
                          accumulation_steps: int) -> tuple[jax.Array, Any]:
    """Sum globally normalized microbatch gradients before clipping or updating the optimizer."""
    local_batch_size = batch['input_ids'].shape[0]
    if accumulation_steps < 1 or local_batch_size % accumulation_steps:
        raise ValueError('Accumulation must evenly partition the fixed local effective batch.')
    if accumulation_steps == 1:
        (_, numerator), gradients = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
        return numerator, gradients
    microbatches = jax.tree.map(lambda value: value.reshape(
        accumulation_steps, local_batch_size // accumulation_steps, *value.shape[1:]), batch)
    initial = (jnp.zeros((), dtype=jnp.float32), jax.tree.map(jnp.zeros_like, parameters))

    def accumulate(carry: tuple[jax.Array, Any], microbatch: dict[str, jax.Array]) -> tuple[tuple[jax.Array, Any], None]:
        """Recompute one microbatch backward graph and add its loss numerator and gradient."""
        total_numerator, total_gradients = carry
        (_, numerator), gradients = jax.value_and_grad(objective, has_aux=True)(parameters, microbatch)
        return (total_numerator + numerator, jax.tree.map(jnp.add, total_gradients, gradients)), None

    (numerator, gradients), _ = jax.lax.scan(accumulate, initial, microbatches)
    return numerator, gradients

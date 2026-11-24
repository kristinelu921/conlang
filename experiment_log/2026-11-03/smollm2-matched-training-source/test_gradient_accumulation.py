"""Compare accumulated token-weighted gradients and optimizer updates with a full batch."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gradient_accumulation import accumulated_gradients
from train_gsm8k import masked_loss_sum


@pytest.mark.parametrize('accumulation_steps', [1, 2, 4])
def test_accumulation_preserves_global_token_weighted_gradient_and_update(accumulation_steps: int) -> None:
    """Unequal masks and an all-padding microbatch must retain the whole-batch objective."""
    parameters = {'embedding': jnp.arange(35, dtype=jnp.float32).reshape(7, 5) / 30,
                  'projection': jnp.arange(35, dtype=jnp.float32).reshape(5, 7) / 50}
    batch = {'input_ids': jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 3, 2], [0, 0, 0, 0]]),
             'loss_mask': jnp.array([[0., 0., 1., 1.], [0., 1., 1., 1.], [0., 0., 0., 1.], [0., 0., 0., 0.]])}
    count = batch['loss_mask'][:, 1:].sum()
    def objective(weights: dict[str, jax.Array], selected: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        """Normalize each microbatch by the original complete batch's supervised-token count."""
        logits = weights['embedding'][selected['input_ids']] @ weights['projection']
        numerator, _ = masked_loss_sum(logits, selected['input_ids'], selected['loss_mask'])
        return numerator / count, numerator
    (_, full_numerator), full_gradients = jax.value_and_grad(objective, has_aux=True)(parameters, batch)
    numerator, gradients = jax.jit(lambda values, rows: accumulated_gradients(values, rows, objective, accumulation_steps))(parameters, batch)
    np.testing.assert_allclose(numerator, full_numerator, rtol=1e-6, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(gradients), jax.tree.leaves(full_gradients)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    optimizer = optax.chain(optax.clip_by_global_norm(.01), optax.adamw(1e-4, weight_decay=.01))
    state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, state, parameters)
    expected_updates, _ = optimizer.update(full_gradients, state, parameters)
    for actual, expected in zip(jax.tree.leaves(updates), jax.tree.leaves(expected_updates)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_invalid_partition_fails_before_execution() -> None:
    """Reject accumulation that changes effective batch coverage."""
    with pytest.raises(ValueError):
        accumulated_gradients({}, {'input_ids': jnp.zeros((4, 3), dtype=jnp.int32)}, lambda weights, rows: (0., 0.), 3)

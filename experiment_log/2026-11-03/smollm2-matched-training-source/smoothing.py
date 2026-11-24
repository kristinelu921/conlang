"""Apply optional target smoothing during training while retaining ordinary evaluation NLL."""

import jax
import jax.numpy as jnp

from train_gsm8k import masked_loss_sum


def smoothed_loss_sum(logits: jax.Array, input_ids: jax.Array, loss_mask: jax.Array,
                      label_smoothing: float) -> tuple[jax.Array, jax.Array]:
    """Mix hard answer targets with uniform labels without allocating dense one-hot tensors."""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("Label smoothing must be in [0, 1).")
    if label_smoothing == 0.0:
        return masked_loss_sum(logits, input_ids, loss_mask)
    hard_sum, token_count = masked_loss_sum(logits, input_ids, loss_mask)
    shifted_logits = logits[:, :-1]
    uniform_loss = jax.nn.logsumexp(shifted_logits, axis=-1) - jnp.mean(shifted_logits, axis=-1)
    uniform_sum = jnp.sum(uniform_loss * loss_mask[:, 1:])
    # When Does Label Smoothing Help? https://arxiv.org/abs/1906.02629
    return (1.0 - label_smoothing) * hard_sum + label_smoothing * uniform_sum, token_count

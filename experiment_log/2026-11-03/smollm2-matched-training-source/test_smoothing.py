"""Compare optional smoothing and its gradients against an independent PyTorch oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from smoothing import smoothed_loss_sum
from train_gsm8k import masked_loss_sum


@pytest.mark.parametrize("smoothing", [0.0, 0.05, 0.1, 0.4])
def test_smoothed_loss_and_gradients_match_pytorch(smoothing: float) -> None:
    """Preserve causal shifting, weighted answer masks, and zero gradients on ignored positions."""
    values = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 7
    tokens = np.array([[1, 3, 0], [0, 2, 4]], dtype=np.int32)
    mask = np.array([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    logits = jnp.asarray(values)
    loss, count = smoothed_loss_sum(logits, jnp.asarray(tokens), jnp.asarray(mask), smoothing)
    gradients = jax.grad(lambda array: smoothed_loss_sum(array, jnp.asarray(tokens), jnp.asarray(mask), smoothing)[0])(logits)
    reference = torch.tensor(values, requires_grad=True)
    reference_losses = torch.nn.functional.cross_entropy(reference[:, :-1].reshape(-1, 5),
        torch.tensor(tokens[:, 1:].reshape(-1), dtype=torch.int64), label_smoothing=smoothing, reduction="none")
    reference_sum = (reference_losses * torch.tensor(mask[:, 1:].reshape(-1))).sum()
    reference_sum.backward()
    np.testing.assert_allclose(loss, reference_sum.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(gradients, reference.grad.numpy(), rtol=1e-6, atol=1e-7)
    assert float(count) == 3.0
    assert not np.asarray(gradients[0, 0]).any()
    if smoothing == 0.0:
        original, original_count = masked_loss_sum(logits, jnp.asarray(tokens), jnp.asarray(mask))
        np.testing.assert_array_equal(loss, original)
        np.testing.assert_array_equal(count, original_count)


@pytest.mark.parametrize("smoothing", [-0.1, 1.0])
def test_invalid_smoothing_is_rejected(smoothing: float) -> None:
    """Reject coefficients that cannot define a valid target distribution."""
    with pytest.raises(ValueError):
        smoothed_loss_sum(jnp.zeros((1, 2, 3)), jnp.zeros((1, 2), dtype=jnp.int32),
                          jnp.ones((1, 2)), smoothing)

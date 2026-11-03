"""Remote CPU differential tests against Hugging Face's independent Qwen3 implementation."""

import unittest
from pathlib import Path

if not Path(__file__).resolve().is_relative_to("/home/kristinelu/peer-tree"):
    raise unittest.SkipTest("Numerical model tests execute only on authorized remote compute.")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_backend import QwenForward, policy_loss


class NumericalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(71)
        self.configuration = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            tie_word_embeddings=True, attention_dropout=0, max_position_embeddings=32, rope_theta=1000000)
        self.reference = Qwen3ForCausalLM(self.configuration).eval()
        self.reference.config._attn_implementation = "eager"
        tensors = {key: value.detach().numpy() for key, value in self.reference.state_dict().items()}
        self.weights = {"embedding": jnp.asarray(tensors["model.embed_tokens.weight"]),
                        "norm": jnp.asarray(tensors["model.norm.weight"]), "layers": []}
        for index in range(2):
            prefix = f"model.layers.{index}."
            layer = {name: jnp.asarray(tensors[prefix + name + ".weight"])
                     for name in ("input_layernorm", "post_attention_layernorm")}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                value = tensors[prefix + "self_attn." + name + ".weight"]
                layer[name] = jnp.asarray(value.T if name.endswith("proj") else value)
            for name in ("gate_proj", "up_proj", "down_proj"):
                layer[name] = jnp.asarray(tensors[prefix + "mlp." + name + ".weight"].T)
            self.weights["layers"].append(layer)
        self.weights["layers"] = tuple(self.weights["layers"])
        random = np.random.default_rng(17)
        self.adapters = tuple({"q_a": jnp.asarray(random.normal(0, .02, (16, 2)), jnp.float32),
                               "q_b": jnp.zeros((2, 16)),
                               "v_a": jnp.asarray(random.normal(0, .02, (16, 2)), jnp.float32),
                               "v_b": jnp.zeros((2, 8))} for _ in range(2))
        self.forward = QwenForward(self.configuration.to_dict(), 2)

    def test_float32_forward_matches_independent_transformers_with_left_padding(self) -> None:
        tokens = np.array([[0, 0, 11, 12, 13], [5, 6, 7, 8, 9]], np.int32)
        mask = np.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], np.int32)
        positions = np.maximum(mask.cumsum(-1) - 1, 0)
        with torch.no_grad():
            expected = self.reference(torch.tensor(tokens, dtype=torch.long),
                attention_mask=torch.tensor(mask), position_ids=torch.tensor(positions)).logits.numpy()
        hidden, _ = self.forward(self.weights, self.adapters, jnp.asarray(tokens), jnp.asarray(mask), jnp.asarray(positions))
        actual = np.asarray(hidden @ self.weights["embedding"].T)
        np.testing.assert_allclose(actual[mask.astype(bool)], expected[mask.astype(bool)], atol=2e-5, rtol=2e-5)

    def test_cached_decode_matches_full_context_forward(self) -> None:
        tokens = jnp.array([[0, 11, 12, 13, 14]], jnp.int32)
        mask = jnp.array([[0, 1, 1, 1, 1]], jnp.int32)
        positions = jnp.maximum(mask.cumsum(-1) - 1, 0)
        expected, _ = self.forward(self.weights, self.adapters, tokens, mask, positions)
        cache = tuple((jnp.zeros((1, 5, 1, 8)), jnp.zeros((1, 5, 1, 8))) for _ in range(2))
        _, cache = self.forward(self.weights, self.adapters, tokens[:, :3], mask, positions[:, :3], cache, jnp.int32(0))
        for index in (3, 4):
            actual, cache = self.forward(self.weights, self.adapters, tokens[:, index:index + 1], mask,
                                         positions[:, index:index + 1], cache, jnp.int32(index))
            np.testing.assert_allclose(np.asarray(actual[:, 0]), np.asarray(expected[:, index]), atol=2e-5, rtol=2e-5)

    def test_adapter_gradient_is_nonzero_and_frozen_weights_are_not_update_arguments(self) -> None:
        def objective(adapters: tuple) -> jax.Array:
            hidden, _ = self.forward(self.weights, adapters, jnp.array([[11, 12, 13]]),
                                     jnp.ones((1, 3), jnp.int32), jnp.array([[0, 1, 2]]))
            logits = hidden[:, -1] @ self.weights["embedding"].T
            return -jax.nn.log_softmax(logits)[0, 17]
        gradients = jax.grad(objective)(self.adapters)
        self.assertGreater(sum(float(jnp.sum(jnp.abs(layer["v_b"]))) for layer in gradients), 0)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree.leaves(gradients)))
        self.assertTrue(all(float(jnp.sum(jnp.abs(layer["q_a"]))) == 0 for layer in gradients))

    def test_success_and_failure_have_opposite_gradients_and_padding_has_none(self) -> None:
        old = jnp.array([[-.7, -.5, -.2]])
        mask = jnp.array([[1., 1., 0.]])
        def objective(log_probabilities: jax.Array, weight: float) -> jax.Array:
            return policy_loss(log_probabilities, old, old, mask, jnp.array([weight]), .2, 0)[0]
        positive = np.asarray(jax.grad(objective)(old, 1.))
        negative = np.asarray(jax.grad(objective)(old, -1.))
        np.testing.assert_allclose(positive, [[-.5, -.5, 0]])
        np.testing.assert_allclose(negative, [[.5, .5, 0]])

    def test_clipping_stops_overreinforcing_a_positive_action(self) -> None:
        old = jnp.array([[-2.]])
        current = jnp.array([[-.5]])
        gradient = jax.grad(lambda value: policy_loss(value, old, old, jnp.ones((1, 1)),
                            jnp.array([1.]), .2, 0)[0])(current)
        np.testing.assert_allclose(gradient, 0)


if __name__ == "__main__":
    unittest.main()

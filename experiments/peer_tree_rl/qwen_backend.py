"""Pinned Qwen3 forward passes, cached TPU decoding, and separate peer LoRA updates."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

from protocol import Generation, State, TrainingItem

ArrayTree = dict[str, Any]


def rms_norm(hidden: jax.Array, weight: jax.Array, epsilon: float) -> jax.Array:
    """Normalize in float32 and restore the activation precision before the scale."""
    normalized = hidden.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.mean(jnp.square(hidden.astype(jnp.float32)), axis=-1, keepdims=True) + epsilon)
    return normalized.astype(hidden.dtype) * weight.astype(hidden.dtype)


def rotary(hidden: jax.Array, positions: jax.Array, theta: float) -> jax.Array:
    """Apply Qwen3's unscaled rotary positions to normalized attention queries and keys."""
    dimension = hidden.shape[-1]
    frequencies = theta ** (-jnp.arange(0, dimension, 2, dtype=jnp.float32) / dimension)
    angles = positions[..., None].astype(jnp.float32) * frequencies
    angles = jnp.concatenate((angles, angles), axis=-1)[..., None, :]
    half = dimension // 2
    rotated = jnp.concatenate((-hidden[..., half:], hidden[..., :half]), axis=-1)
    return hidden * jnp.cos(angles).astype(hidden.dtype) + rotated * jnp.sin(angles).astype(hidden.dtype)


class QwenForward:
    """Implement the pinned dense Qwen3 architecture with frozen weights and q/v adapters."""

    # Qwen3 Technical Report https://arxiv.org/abs/2505.09388; preserve per-head QK normalization and checkpoint RoPE.

    def __init__(self, configuration: dict, scale: float) -> None:
        self.configuration = configuration
        self.scale = scale

    def layer(self, weights: ArrayTree, adapter: ArrayTree, hidden: jax.Array,
              mask: jax.Array, positions: jax.Array, cache: tuple | None,
              cache_index: jax.Array) -> tuple[jax.Array, tuple]:
        """Apply one pre-normalized attention/MLP layer, optionally updating its KV cache."""
        configuration = self.configuration
        normalized = rms_norm(hidden, weights["input_layernorm"], configuration["rms_norm_eps"])
        query = normalized @ weights["q_proj"] + self.scale * (
            normalized.astype(jnp.float32) @ adapter["q_a"] @ adapter["q_b"]).astype(hidden.dtype)
        key = normalized @ weights["k_proj"]
        value = normalized @ weights["v_proj"] + self.scale * (
            normalized.astype(jnp.float32) @ adapter["v_a"] @ adapter["v_b"]).astype(hidden.dtype)
        batch, length = hidden.shape[:2]
        head_dimension = configuration["head_dim"]
        query = query.reshape(batch, length, configuration["num_attention_heads"], head_dimension)
        key = key.reshape(batch, length, configuration["num_key_value_heads"], head_dimension)
        value = value.reshape(key.shape)
        query = rotary(rms_norm(query, weights["q_norm"], configuration["rms_norm_eps"]),
                       positions, configuration["rope_theta"])
        key = rotary(rms_norm(key, weights["k_norm"], configuration["rms_norm_eps"]),
                     positions, configuration["rope_theta"])
        if cache is not None:
            key = jax.lax.dynamic_update_slice(cache[0], key, (0, cache_index, 0, 0))
            value = jax.lax.dynamic_update_slice(cache[1], value, (0, cache_index, 0, 0))
        new_cache = (key, value)
        groups = configuration["num_attention_heads"] // configuration["num_key_value_heads"]
        key = jnp.repeat(key, groups, axis=2)
        value = jnp.repeat(value, groups, axis=2)
        scores = jnp.einsum("bqhd,bkhd->bhqk", query, key).astype(jnp.float32) / np.sqrt(head_dimension)
        physical_queries = cache_index + jnp.arange(length)
        allowed = (jnp.arange(key.shape[1])[None, :] <= physical_queries[:, None])[None, None, :, :]
        allowed = allowed & mask[:, None, None, :].astype(bool)
        scores = jnp.where(allowed, scores, jnp.finfo(jnp.float32).min)
        probabilities = jax.nn.softmax(scores, axis=-1).astype(hidden.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value).reshape(batch, length, -1)
        hidden = hidden + attended @ weights["o_proj"]
        normalized = rms_norm(hidden, weights["post_attention_layernorm"], configuration["rms_norm_eps"])
        feedforward = jax.nn.silu(normalized @ weights["gate_proj"]) * (normalized @ weights["up_proj"])
        return hidden + feedforward @ weights["down_proj"], new_cache

    def __call__(self, weights: ArrayTree, adapters: tuple, tokens: jax.Array,
                 mask: jax.Array, positions: jax.Array, cache: tuple | None = None,
                 cache_index: jax.Array = 0) -> tuple[jax.Array, tuple]:
        """Return normalized hidden states, projecting the vocabulary only where needed."""
        hidden = weights["embedding"][tokens]
        caches = []
        layer_function = jax.checkpoint(self.layer) if cache is None else self.layer
        for index, layer_weights in enumerate(weights["layers"]):
            hidden, layer_cache = layer_function(layer_weights, adapters[index], hidden, mask,
                positions, None if cache is None else cache[index], cache_index)
            if cache is not None:
                caches.append(layer_cache)
        return rms_norm(hidden, weights["norm"], self.configuration["rms_norm_eps"]), tuple(caches)


def token_log_probabilities(logits: jax.Array, targets: jax.Array, temperature: float) -> jax.Array:
    """Score the exact full-support tempered policy used by rollout sampling."""
    scaled = logits.astype(jnp.float32) / temperature
    return jnp.take_along_axis(scaled, targets[..., None], axis=-1)[..., 0] - jax.nn.logsumexp(scaled, axis=-1)


def policy_loss(log_probabilities: jax.Array, old_log_probabilities: jax.Array,
                reference_log_probabilities: jax.Array, mask: jax.Array, weights: jax.Array,
                ratio_clip: float, kl_coefficient: float) -> tuple[jax.Array, dict]:
    """Compute clipped per-message policy credit and frozen-base sampled KL regularization."""
    ratios = jnp.exp(jnp.clip(log_probabilities - old_log_probabilities, -20, 20))
    advantages = jax.lax.stop_gradient(weights[:, None])
    surrogate = jnp.minimum(ratios * advantages, jnp.clip(ratios, 1 - ratio_clip, 1 + ratio_clip) * advantages)
    difference = jnp.clip(reference_log_probabilities - log_probabilities, -20, 20)
    divergence = jnp.exp(difference) - difference - 1
    token_count = jnp.maximum(mask.sum(axis=-1), 1)
    valid = (mask.sum(axis=-1) > 0).astype(jnp.float32)
    message_loss = ((-surrogate + kl_coefficient * divergence) * mask).sum(axis=-1) / token_count
    return (message_loss * valid).sum(), {"valid": valid.sum(),
        "kl_sum": ((divergence * mask).sum(axis=-1) / token_count * valid).sum(),
        "log_probability_sum": ((log_probabilities * mask).sum(axis=-1) / token_count * valid).sum()}


class QwenBackend:
    """Own one frozen base and two independent LoRA/optimizer states on a single TPU VM."""

    def __init__(self, settings: dict, cache_directory: Path) -> None:
        self.settings = settings
        jax.config.update("jax_default_matmul_precision", "highest")
        self.devices = jax.local_devices(backend="tpu")
        if len(self.devices) != 8 or jax.process_count() != 1:
            raise RuntimeError("This pilot requires one authorized host with eight TPU devices.")
        self.batch_size = len(self.devices)
        self.prompt_width = settings["maximum_prompt_tokens"]
        self.output_width = settings["maximum_new_tokens"]
        self.total_width = self.prompt_width + self.output_width
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["model_revision"],
                                                       cache_dir=str(cache_directory))
        configuration = AutoConfig.from_pretrained(settings["model"], revision=settings["model_revision"],
                                                   cache_dir=str(cache_directory)).to_dict()
        if (configuration["model_type"] != "qwen3" or not configuration["tie_word_embeddings"]
                or configuration.get("rope_scaling") or configuration.get("use_sliding_window")
                or configuration["attention_bias"] or configuration["attention_dropout"] != 0):
            raise ValueError("Checkpoint architecture is outside this verified dense-Qwen3 adapter.")
        self.configuration = configuration
        self.forward = QwenForward(configuration, settings["lora_alpha"] / settings["lora_rank"])
        self.snapshot = Path(snapshot_download(settings["model"], revision=settings["model_revision"],
            cache_dir=str(cache_directory), allow_patterns=["*.safetensors", "*.json", "*.txt"]))
        host_weights = self.load_weights()
        with jax.default_device(jax.devices("cpu")[0]):
            host_weights = jax.tree.map(lambda value: jnp.asarray(value, dtype=jnp.float32), host_weights)
        self.weights = jax.device_put_replicated(host_weights, self.devices)
        self.states = []
        optimizer = optax.chain(optax.clip_by_global_norm(settings["gradient_clip_norm"]),
            optax.adam(settings["learning_rate"], b1=settings["adam_beta1"], b2=settings["adam_beta2"]))
        self.initial_adapters = []
        for agent in range(2):
            random = np.random.default_rng(settings["seed"] + agent)
            adapters = tuple({"q_a": random.normal(0, 0.02, (configuration["hidden_size"], settings["lora_rank"])).astype("float32"),
                              "q_b": np.zeros((settings["lora_rank"], configuration["num_attention_heads"] * configuration["head_dim"]), "float32"),
                              "v_a": random.normal(0, 0.02, (configuration["hidden_size"], settings["lora_rank"])).astype("float32"),
                              "v_b": np.zeros((settings["lora_rank"], configuration["num_key_value_heads"] * configuration["head_dim"]), "float32")}
                             for _ in range(configuration["num_hidden_layers"]))
            with jax.default_device(jax.devices("cpu")[0]):
                state = TrainState.create(apply_fn=self.forward, params=adapters, tx=optimizer)
            self.states.append(jax.device_put_replicated(state, self.devices))
            self.initial_adapters.append(jax.device_put_replicated(adapters, self.devices))
        self.generate_parallel = jax.pmap(self.generate_device, devices=self.devices)
        self.update_parallel = jax.pmap(self.update_device, axis_name="devices", devices=self.devices,
                                       donate_argnums=(1,))
        self.log_probability_parallel = jax.pmap(self.score_device, devices=self.devices)

    def load_weights(self) -> ArrayTree:
        """Convert only the pinned checkpoint tensors and reject missing or unexpected keys."""
        flat = {}
        for path in sorted(self.snapshot.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    flat[key] = handle.get_tensor(key).float().numpy()
        weights: ArrayTree = {"embedding": flat.pop("model.embed_tokens.weight"),
                              "norm": flat.pop("model.norm.weight"), "layers": []}
        for index in range(self.configuration["num_hidden_layers"]):
            layer = {}
            prefix = f"model.layers.{index}."
            for name in ("input_layernorm", "post_attention_layernorm"):
                layer[name] = flat.pop(prefix + name + ".weight")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
                array = flat.pop(prefix + "self_attn." + name + ".weight")
                layer[name] = array.T if name.endswith("proj") else array
            for name in ("gate_proj", "up_proj", "down_proj"):
                layer[name] = flat.pop(prefix + "mlp." + name + ".weight").T
            weights["layers"].append(layer)
        if "lm_head.weight" in flat:
            if not np.array_equal(flat.pop("lm_head.weight"), weights["embedding"]):
                raise ValueError("The checkpoint claims tied embeddings but provides a different head.")
        if flat:
            raise ValueError(f"Unused checkpoint weights: {sorted(flat)}")
        weights["layers"] = tuple(weights["layers"])
        return weights

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Keep all question/context tokens and insert Qwen's explicit no-thinking template."""
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False)
        assert isinstance(encoded, list)
        if not encoded or len(encoded) > self.prompt_width:
            raise ValueError(f"Full prompt exceeds the frozen width: {len(encoded)} > {self.prompt_width}")
        return encoded

    def generate_device(self, weights: ArrayTree, adapters: tuple, tokens: jax.Array,
                        mask: jax.Array, seed: jax.Array) -> tuple:
        """Decode a fixed-width batch with cached keys/values and stored behavior log probabilities."""
        configuration = self.configuration
        shape = (tokens.shape[0], self.total_width, configuration["num_key_value_heads"], configuration["head_dim"])
        cache = tuple((jnp.zeros(shape, weights["embedding"].dtype), jnp.zeros(shape, weights["embedding"].dtype))
                      for _ in range(configuration["num_hidden_layers"]))
        extended_mask = jnp.concatenate((mask, jnp.ones((tokens.shape[0], self.output_width), jnp.int32)), axis=1)
        positions = jnp.maximum(mask.cumsum(axis=-1) - 1, 0)
        hidden, cache = self.forward(weights, adapters, tokens, extended_mask, positions, cache, jnp.int32(0))
        logits = hidden[:, -1] @ weights["embedding"].T
        output = jnp.full((tokens.shape[0], self.output_width), self.tokenizer.pad_token_id, jnp.int32)
        log_probabilities = jnp.zeros(output.shape, jnp.float32)
        random_keys = jax.vmap(jax.random.PRNGKey)(seed)
        initial = (jnp.int32(0), cache, logits, output, log_probabilities,
                   jnp.zeros((tokens.shape[0],), bool), random_keys)

        def condition(carry: tuple) -> jax.Array:
            return (carry[0] < self.output_width) & ~jnp.all(carry[5])

        def step(carry: tuple) -> tuple:
            index, current_cache, current_logits, generated, probabilities, finished, keys = carry
            split_keys = jax.vmap(jax.random.split)(keys)
            keys, sampling_keys = split_keys[:, 0], split_keys[:, 1]
            scaled = current_logits.astype(jnp.float32) / self.settings["temperature"]
            sampled = jax.vmap(jax.random.categorical)(sampling_keys, scaled).astype(jnp.int32)
            selected = token_log_probabilities(current_logits, sampled, self.settings["temperature"])
            sampled = jnp.where(finished, self.tokenizer.pad_token_id, sampled)
            generated = generated.at[:, index].set(sampled)
            probabilities = probabilities.at[:, index].set(jnp.where(finished, 0, selected))
            finished = finished | (sampled == self.tokenizer.eos_token_id) | (sampled == 151643)
            next_positions = positions[:, -1:] + index + 1
            hidden, current_cache = self.forward(weights, adapters, sampled[:, None], extended_mask,
                next_positions, current_cache, self.prompt_width + index)
            next_logits = hidden[:, -1] @ weights["embedding"].T
            return index + 1, current_cache, next_logits, generated, probabilities, finished, keys

        result = jax.lax.while_loop(condition, step, initial)
        return result[3], result[4], result[5]

    def generate(self, states: list[State], seeds: list[int], agent: int) -> list[Generation]:
        """Batch independent message actions without exposing training answers to the policy."""
        return self.generate_messages([state.messages() for state in states], seeds, agent)

    def generate_messages(self, messages: list[list[dict[str, str]]], seeds: list[int], agent: int) -> list[Generation]:
        """Generate padded TPU batches while retaining exact unpadded replay tokens."""
        if len(messages) != len(seeds):
            raise ValueError("Every generated item requires an independent sampling seed.")
        results = []
        for start in range(0, len(messages), self.batch_size):
            prompts = [self.encode_messages(item) for item in messages[start:start + self.batch_size]]
            tokens = np.full((self.batch_size, 1, self.prompt_width), self.tokenizer.pad_token_id, np.int32)
            mask = np.zeros(tokens.shape, np.int32)
            batch_seeds = np.zeros((self.batch_size, 1), np.uint32)
            for row, prompt in enumerate(prompts):
                tokens[row, 0, -len(prompt):] = prompt
                mask[row, 0, -len(prompt):] = 1
                batch_seeds[row, 0] = seeds[start + row]
            # Dummy lanes use a valid prompt and independent key, and never enter artifacts or updates.
            for row in range(len(prompts), self.batch_size):
                tokens[row] = tokens[0]
                mask[row] = mask[0]
                batch_seeds[row, 0] = np.uint32(row)
            generated, probabilities, ended = jax.device_get(self.generate_parallel(
                self.weights, self.states[agent].params, tokens, mask, batch_seeds))
            for row, prompt in enumerate(prompts):
                output = generated[row, 0].tolist()
                stop_indices = [index for index, token in enumerate(output)
                                if token in (self.tokenizer.eos_token_id, 151643)]
                length = stop_indices[0] + 1 if stop_indices else self.output_width
                output = output[:length]
                results.append(Generation(tuple(prompt), tuple(output), tuple(probabilities[row, 0, :length].tolist()),
                    self.tokenizer.decode(output, skip_special_tokens=True), bool(ended[row, 0])))
        return results

    def score_device(self, weights: ArrayTree, adapters: tuple, tokens: jax.Array,
                     mask: jax.Array) -> jax.Array:
        """Score response tokens only, while attending to the entire masked-loss prompt."""
        positions = jnp.maximum(mask.cumsum(axis=-1) - 1, 0)
        hidden, _ = self.forward(weights, adapters, tokens, mask, positions)
        logits = hidden[:, self.prompt_width - 1:-1] @ weights["embedding"].T
        return token_log_probabilities(logits, tokens[:, self.prompt_width:], self.settings["temperature"])

    def update_device(self, weights: ArrayTree, state: TrainState, reference: tuple, batch: dict) -> tuple:
        """Update only this peer's adapter using globally normalized distinct-message credit."""
        reference_probabilities = self.score_device(weights, reference, batch["tokens"], batch["attention_mask"])
        valid_count = jax.lax.psum((batch["completion_mask"].sum(axis=-1) > 0).sum(), "devices")

        def objective(adapters: tuple) -> tuple:
            probabilities = self.score_device(weights, adapters, batch["tokens"], batch["attention_mask"])
            loss, metrics = policy_loss(probabilities, batch["old_log_probabilities"],
                jax.lax.stop_gradient(reference_probabilities), batch["completion_mask"], batch["weights"],
                self.settings["ratio_clip"], self.settings["reference_kl_coefficient"])
            return loss / jnp.maximum(valid_count, 1), metrics

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        gradients = jax.lax.psum(gradients, "devices")
        metrics = {key: jax.lax.psum(value, "devices") for key, value in metrics.items()}
        metrics.update(loss=jax.lax.psum(loss, "devices"), gradient_norm=optax.global_norm(gradients))
        return state.apply_gradients(grads=gradients), metrics

    def training_batch(self, items: list[TrainingItem]) -> dict[str, np.ndarray]:
        """Pad complete prompt/response pairs with loss restricted to sampled outgoing tokens."""
        if len(items) > self.batch_size:
            raise ValueError("Training microbatch exceeds the fixed eight-device shape.")
        batch = {"tokens": np.full((self.batch_size, 1, self.total_width), self.tokenizer.pad_token_id, np.int32),
                 "attention_mask": np.zeros((self.batch_size, 1, self.total_width), np.int32),
                 "completion_mask": np.zeros((self.batch_size, 1, self.output_width), np.float32),
                 "old_log_probabilities": np.zeros((self.batch_size, 1, self.output_width), np.float32),
                 "weights": np.zeros((self.batch_size, 1), np.float32)}
        for row, item in enumerate(items):
            output = item.generation
            if self.encode_messages(item.state.messages()) != list(output.prompt_ids):
                raise ValueError("Replay prompt differs from the exact generation context.")
            prompt_start = self.prompt_width - len(output.prompt_ids)
            output_end = self.prompt_width + len(output.token_ids)
            if prompt_start < 0 or output_end > self.total_width:
                raise ValueError("Replay must not truncate a prompt or generated response.")
            batch["tokens"][row, 0, prompt_start:self.prompt_width] = output.prompt_ids
            batch["tokens"][row, 0, self.prompt_width:output_end] = output.token_ids
            batch["attention_mask"][row, 0, prompt_start:output_end] = 1
            batch["completion_mask"][row, 0, :len(output.token_ids)] = 1
            batch["old_log_probabilities"][row, 0, :len(output.token_ids)] = output.log_probabilities
            batch["weights"][row, 0] = item.weight
        return batch

    def update(self, items: list[TrainingItem], check_time: Any) -> list[dict]:
        """Apply one replay epoch separately for each peer, with no gradient across sampled text."""
        metrics = []
        for agent in range(2):
            selected = [item for item in items if item.agent == agent]
            for start in range(0, len(selected), self.batch_size):
                check_time()
                batch = self.training_batch(selected[start:start + self.batch_size])
                beginning = time.monotonic()
                self.states[agent], values = self.update_parallel(self.weights, self.states[agent],
                    self.initial_adapters[agent], batch)
                host_values = jax.device_get(values)
                row = {key: float(value[0]) for key, value in host_values.items()}
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError("Nonfinite policy update; no checkpoint accepted.")
                row.update(agent=agent, elapsed_seconds=time.monotonic() - beginning,
                           optimizer_step=int(jax.device_get(self.states[agent].step)[0]))
                metrics.append(row)
        return metrics

    def verify_replay(self, items: list[TrainingItem]) -> dict:
        """Check cached rollout probabilities against teacher-forced replay before any update."""
        differences = []
        for agent in range(2):
            selected = [item for item in items if item.agent == agent]
            for start in range(0, len(selected), self.batch_size):
                batch = self.training_batch(selected[start:start + self.batch_size])
                replayed = np.asarray(self.log_probability_parallel(self.weights, self.states[agent].params,
                    batch["tokens"], batch["attention_mask"]))
                selected_mask = batch["completion_mask"].astype(bool)
                differences.extend(np.abs(replayed - batch["old_log_probabilities"])[selected_mask].tolist())
        maximum = max(differences)
        mean = float(np.mean(differences))
        if maximum > 0.1 or mean > 0.02:
            raise ValueError(f"Cached/replay policy mismatch: maximum={maximum}, mean={mean}")
        return {"maximum_log_probability_difference": maximum, "mean_log_probability_difference": mean,
                "compared_tokens": len(differences)}

    def checkpoint(self, directory: Path) -> None:
        """Serialize both adapter/optimizer states without copying the frozen pretrained base."""
        directory.mkdir(parents=True, exist_ok=False)
        for agent, state in enumerate(self.states):
            host_state = jax.tree.map(lambda array: np.asarray(array.addressable_shards[0].data).squeeze(0), state)
            (directory / f"agent-{agent}.msgpack").write_bytes(serialization.to_bytes(host_state))
        (directory / "configuration.json").write_text(json.dumps(self.settings, indent=2) + "\n")

    def restore(self, directory: Path) -> None:
        """Restore both peers including Adam moments and step counters from a verified checkpoint."""
        if json.loads((directory / "configuration.json").read_text()) != self.settings:
            raise ValueError("Resume configuration differs from the frozen pilot.")
        restored = []
        for agent, state in enumerate(self.states):
            template = jax.tree.map(lambda array: np.asarray(array.addressable_shards[0].data).squeeze(0), state)
            value = serialization.from_bytes(template, (directory / f"agent-{agent}.msgpack").read_bytes())
            restored.append(jax.device_put_replicated(value, self.devices))
        self.states = restored

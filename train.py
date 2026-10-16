"""Train the linked-facts communication model with JAX on a single-host TPU."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import jax_utils, serialization
from flax.training.train_state import TrainState

from jax_models import JaxCommunicationModel, JaxCommunicationModelConfig, JaxCommunicationOutput
from linked_facts.generate_dataset import verify_dataset
from tokenizer import TextTokenizer, build_task_tokenizer
from regional_data import DATASET_PREFIX, resolve_training_dataset
from training_data import load_examples, resolve_split_path


Array = jax.Array
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_node_local_path(path: Path, directory_name: str) -> bool:
    """Return whether a path is below /home/<user>/<directory_name>/."""
    parts = path.resolve(strict=False).parts
    return (
        path.is_absolute()
        and len(parts) >= 5
        and parts[0] == "/"
        and parts[1] == "home"
        and bool(parts[2])
        and parts[3] == directory_name
    )


@dataclass(frozen=True)
class TrainConfig:
    """Complete configuration for one JAX communication-model run."""

    run_id: str = "conlang-jax-local"
    dataset_dir: Path | None = None
    dataset_cache_dir: Path | None = None
    dataset_prefix: str = DATASET_PREFIX
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    steps: int = 1000
    eval_every: int = 100
    eval_examples: int = 256
    log_every: int = 10
    throughput_warmup_steps: int = 10
    seed: int = 1729
    device: str = "cpu"
    precision: str = "float32"
    output_path: Path | None = None
    limit_examples: int | None = None
    repeat_training_examples_to_batch_size: bool = False
    evaluation_message_interventions: bool = False
    wandb_project: str = "conlangs"
    wandb_run_name: str | None = None
    wandb_enabled: bool = True
    hidden_size: int = 128
    output_hidden_size: int | None = None
    tie_sender_receiver_encoder: bool = False
    role_gate_channels: int | None = None
    sender_question_conditioning: bool = False
    message_length: int = 4
    max_answer_length: int = 4
    max_text_length: int = 512
    max_question_length: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

    def validate(self) -> None:
        """Reject invalid science, topology, and node-local path settings."""
        positive_integers = (
            self.batch_size,
            self.steps,
            self.eval_every,
            self.eval_examples,
            self.log_every,
            self.hidden_size,
            self.message_length,
            self.max_answer_length,
            self.max_text_length,
            self.max_question_length,
            self.num_layers,
            self.num_heads,
        )
        if min(positive_integers) < 1:
            raise ValueError("batch, step, evaluation, logging, and model sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.max_gradient_norm <= 0:
            raise ValueError(
                "learning rate and gradient clip must be positive; weight decay cannot be negative"
            )
        if self.throughput_warmup_steps < 1 or self.throughput_warmup_steps >= self.steps:
            raise ValueError("throughput warmup must be positive and shorter than the run")
        if self.limit_examples is not None and self.limit_examples < 1:
            raise ValueError("limit_examples must be positive when configured")
        if self.eval_examples % self.batch_size != 0:
            raise ValueError("eval_examples must be divisible by batch_size for static evaluation")
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id contains unsupported characters")
        if self.device not in {"cpu", "tpu"}:
            raise ValueError("device must be cpu or tpu")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if not self.dataset_prefix or self.dataset_prefix.startswith(("gs://", "/")):
            raise ValueError("dataset prefix must be an object path inside the regional bucket")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden size must be divisible by the number of attention heads")
        if self.output_hidden_size is not None and self.output_hidden_size < 1:
            raise ValueError("output hidden size must be positive when configured")
        if self.role_gate_channels is not None and self.role_gate_channels < 1:
            raise ValueError("role gate channels must be positive when configured")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.relative_attention_num_buckets < 4
            or self.relative_attention_num_buckets % 2 != 0
        ):
            raise ValueError("relative attention needs an even bucket count of at least 4")
        if self.relative_attention_max_distance <= self.relative_attention_num_buckets // 2:
            raise ValueError("relative attention max distance must exceed half the bucket count")
        if self.device == "tpu":
            if self.dataset_dir is not None:
                raise ValueError("TPU runs must resolve the region-matched dataset rather than bypassing it")
            if self.dataset_cache_dir is None or not _is_node_local_path(
                self.dataset_cache_dir, "elf-data"
            ):
                raise ValueError("TPU dataset_cache_dir must be node-local under /home/<user>/elf-data/")
            if self.output_path is None or not _is_node_local_path(self.output_path, "elf-runs"):
                raise ValueError("TPU output_path must be node-local under /home/<user>/elf-runs/")
            if self.output_path.parent.name != self.run_id:
                raise ValueError("TPU output directory must exactly match run_id")


class JaxExampleBatch(NamedTuple):
    """One fixed-shape NumPy batch accepted by JAX."""

    answer_token_ids: np.ndarray
    agent_1_token_ids: np.ndarray
    agent_2_token_ids: np.ndarray
    question_token_ids: np.ndarray


@dataclass(frozen=True)
class TokenizedExamples:
    """Fixed-shape token arrays cached once on the host."""

    answer_token_ids: np.ndarray
    agent_1_token_ids: np.ndarray
    agent_2_token_ids: np.ndarray
    question_token_ids: np.ndarray

    @property
    def size(self) -> int:
        """Return the number of tokenized examples."""
        return int(self.answer_token_ids.shape[0])

    def select(self, indices: Sequence[int]) -> JaxExampleBatch:
        """Gather one fixed-shape batch by example index."""
        index_array = np.asarray(indices, dtype=np.int64)
        return JaxExampleBatch(
            answer_token_ids=self.answer_token_ids[index_array],
            agent_1_token_ids=self.agent_1_token_ids[index_array],
            agent_2_token_ids=self.agent_2_token_ids[index_array],
            question_token_ids=self.question_token_ids[index_array],
        )


def _pad_token_ids(sequences: Sequence[Sequence[int]], padded_length: int, padding_id: int) -> np.ndarray:
    """Pad integer sequences once on the host to one static TPU shape."""
    result = np.full((len(sequences), padded_length), padding_id, dtype=np.int32)
    for row_index, sequence in enumerate(sequences):
        if len(sequence) > padded_length:
            raise ValueError(f"sequence length {len(sequence)} exceeds configured maximum {padded_length}")
        result[row_index, : len(sequence)] = sequence
    return result


def tokenize_examples(
    examples: Sequence[Mapping[str, str]],
    tokenizer: TextTokenizer,
    config: TrainConfig,
) -> TokenizedExamples:
    """Tokenize and statically pad a dataset once before compiled training."""
    answers = [tokenizer.encode_answer(example["answer"]) for example in examples]
    agent_1_inputs = [tokenizer.encode_text(example["agent_1_input"]) for example in examples]
    agent_2_inputs = [tokenizer.encode_text(example["agent_2_input"]) for example in examples]
    questions = [tokenizer.encode_text(example["question"]) for example in examples]
    return TokenizedExamples(
        answer_token_ids=_pad_token_ids(answers, config.max_answer_length, tokenizer.pad_id),
        agent_1_token_ids=_pad_token_ids(agent_1_inputs, config.max_text_length, tokenizer.pad_id),
        agent_2_token_ids=_pad_token_ids(agent_2_inputs, config.max_text_length, tokenizer.pad_id),
        question_token_ids=_pad_token_ids(questions, config.max_question_length, tokenizer.pad_id),
    )


def iter_epoch_batches(
    examples: TokenizedExamples,
    batch_size: int,
    random_generator: random.Random,
    repeat_examples_to_batch_size: bool = False,
) -> Iterator[JaxExampleBatch]:
    """Yield shuffled, full, fixed-shape batches indefinitely."""
    if examples.size < 1:
        raise ValueError("training dataset is empty")
    if examples.size < batch_size and not repeat_examples_to_batch_size:
        raise ValueError("training dataset is smaller than one full batch")
    while True:
        order = list(range(examples.size))
        random_generator.shuffle(order)
        if examples.size < batch_size:
            repeated_order = [order[index % examples.size] for index in range(batch_size)]
            yield examples.select(repeated_order)
            continue
        for start in range(0, examples.size - batch_size + 1, batch_size):
            yield examples.select(order[start : start + batch_size])


def _model_config(config: TrainConfig, tokenizer: TextTokenizer) -> JaxCommunicationModelConfig:
    """Build the Flax model configuration from the effective run settings."""
    dtype = jnp.bfloat16 if config.precision == "bfloat16" else jnp.float32
    return JaxCommunicationModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
        output_hidden_size=config.output_hidden_size,
        tie_sender_receiver_encoder=config.tie_sender_receiver_encoder,
        role_gate_channels=config.role_gate_channels,
        sender_question_conditioning=config.sender_question_conditioning,
        message_length=config.message_length,
        max_answer_length=config.max_answer_length,
        max_text_length=config.max_text_length,
        max_question_length=config.max_question_length,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        padding_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        relative_attention_num_buckets=config.relative_attention_num_buckets,
        relative_attention_max_distance=config.relative_attention_max_distance,
        dtype=dtype,
    )


def _initialize_state(
    model: JaxCommunicationModel,
    optimizer: optax.GradientTransformation,
    batch: JaxExampleBatch,
    seed: int,
) -> TrainState:
    """Initialize model parameters and optimizer state from one example shape."""
    parameter_key, dropout_key = jax.random.split(jax.random.PRNGKey(seed))
    variables = model.init(
        {"params": parameter_key, "dropout": dropout_key},
        jnp.asarray(batch.agent_1_token_ids[:1]),
        jnp.asarray(batch.agent_2_token_ids[:1]),
        jnp.asarray(batch.question_token_ids[:1]),
        jnp.asarray(batch.answer_token_ids[:1]),
        deterministic=False,
    )
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)


def _optimizer_for_params(config: TrainConfig, params: Any) -> optax.GradientTransformation:
    """Create clipped AdamW with weight decay only on matrix parameters."""
    decay_mask = jax.tree_util.tree_map(lambda parameter: parameter.ndim >= 2, params)
    return optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay, mask=decay_mask),
    )


ACTIVATION_HISTOGRAM_EDGES = tuple(float(edge) for edge in np.linspace(-5.0, 5.0, 33))
TOP_VOCABULARY_HISTOGRAM_SIZE = 100


def _activation_histogram_counts(values: Array) -> Array:
    """Count activation values into fixed bins from -5 to 5."""
    edges = jnp.asarray(ACTIVATION_HISTOGRAM_EDGES)
    flat = values.astype(jnp.float32).ravel()
    indexes = jnp.digitize(flat, edges[1:-1])
    indexes = jnp.clip(indexes, 0, edges.shape[0] - 2)
    return jnp.bincount(indexes, length=edges.shape[0] - 1)


def _global_root_mean_square(values: Array) -> Array:
    """Return the root mean square after summing squares across devices."""
    flat = values.astype(jnp.float32).ravel()
    sum_squares = jax.lax.psum(jnp.sum(jnp.square(flat)), axis_name="devices")
    count = jax.lax.psum(jnp.asarray(flat.size, dtype=jnp.float32), axis_name="devices")
    return jnp.sqrt(sum_squares / count)


def _json_metric_value(value: Any) -> Any:
    """Convert one device metric into a JSON-ready scalar or list."""
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


def _top_vocabulary_histogram(
    generated_token_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a token-ID histogram for the 100 most frequent generated tokens."""
    token_id_array = np.asarray(generated_token_ids, dtype=np.int64)
    assert token_id_array.ndim == 2 and token_id_array.size > 0
    unique_token_ids, token_counts = np.unique(token_id_array, return_counts=True)
    frequency_order = np.lexsort((unique_token_ids, -token_counts))
    selected_indexes = frequency_order[:TOP_VOCABULARY_HISTOGRAM_SIZE]
    selected_token_ids = unique_token_ids[selected_indexes]
    selected_counts = token_counts[selected_indexes]
    token_id_order = np.argsort(selected_token_ids)
    selected_token_ids = selected_token_ids[token_id_order]
    selected_counts = selected_counts[token_id_order]
    token_id_values = selected_token_ids.astype(np.float64)
    if token_id_values.size == 1:
        histogram_edges = np.array(
            [token_id_values[0] - 0.5, token_id_values[0] + 0.5],
            dtype=np.float64,
        )
    else:
        midpoints = (token_id_values[:-1] + token_id_values[1:]) / 2.0
        histogram_edges = np.concatenate(
            ([token_id_values[0] - 0.5], midpoints, [token_id_values[-1] + 0.5])
        )
    return selected_token_ids, selected_counts, histogram_edges


def _host_training_record(
    step: int,
    host_metrics: Mapping[str, Any],
    examples_per_second: float | None,
    generated_token_ids: np.ndarray,
) -> dict[str, Any]:
    """Build the scalar and histogram record saved for one logged step."""
    record: dict[str, Any] = {
        "step": float(step),
        "train_loss": float(host_metrics["loss"]),
        "gradient_norm": float(host_metrics["gradient_norm"]),
    }
    if examples_per_second is not None:
        record["examples_per_second"] = examples_per_second
    for key, value in host_metrics.items():
        if key in {"loss", "gradient_norm", "generated_token_ids"}:
            continue
        record[key] = _json_metric_value(value)
    if "message_1_histogram" in record:
        record["activation_histogram_edges"] = list(ACTIVATION_HISTOGRAM_EDGES)
    token_ids, token_counts, histogram_edges = _top_vocabulary_histogram(
        generated_token_ids
    )
    record["generated_vocabulary_top_token_ids"] = token_ids.tolist()
    record["generated_vocabulary_histogram"] = token_counts.tolist()
    record["generated_vocabulary_histogram_edges"] = histogram_edges.tolist()
    return record


def _as_wandb_log(record: Mapping[str, Any]) -> dict[str, Any]:
    """Turn compact histogram lists into W&B histograms and keep scalar metrics."""
    import wandb

    activation_edges = np.asarray(
        record.get("activation_histogram_edges", ACTIVATION_HISTOGRAM_EDGES),
        dtype=np.float64,
    )
    vocabulary_edges = record.get("generated_vocabulary_histogram_edges")
    payload: dict[str, Any] = {}
    for key, value in record.items():
        if key in {
            "activation_histogram_edges",
            "generated_vocabulary_histogram_edges",
            "generated_vocabulary_top_token_ids",
        }:
            continue
        if key == "generated_vocabulary_histogram":
            vocabulary_counts = np.asarray(value)
            if vocabulary_edges is None:
                raise ValueError("generated vocabulary histogram edges are required")
            if vocabulary_counts.size > TOP_VOCABULARY_HISTOGRAM_SIZE:
                raise ValueError("generated vocabulary histogram exceeds the 100-token limit")
            payload[key] = wandb.Histogram(
                np_histogram=(vocabulary_counts, np.asarray(vocabulary_edges, dtype=np.float64))
            )
            continue
        if key.endswith("_histogram"):
            payload[key] = wandb.Histogram(
                np_histogram=(np.asarray(value), activation_edges)
            )
            continue
        payload[key] = value
    return payload


def _unshard_token_ids(token_ids: Array) -> np.ndarray:
    """Flatten a pmapped token array back to host batch order."""
    array = np.asarray(jax.device_get(token_ids))
    return array.reshape(-1, array.shape[-1])


def _append_model_outputs(
    outputs_path: Path,
    step: int,
    generated_token_ids: np.ndarray,
    target_token_ids: np.ndarray,
    tokenizer: TextTokenizer,
) -> None:
    """Append one step's greedy answers and targets to the outputs file."""
    generated_rows = np.asarray(generated_token_ids, dtype=np.int32)
    target_rows = np.asarray(target_token_ids, dtype=np.int32)
    record = {
        "step": step,
        "generated_token_ids": generated_rows.tolist(),
        "target_token_ids": target_rows.tolist(),
        "generated_text": [tokenizer.decode_answer(row) for row in generated_rows.tolist()],
        "target_text": [tokenizer.decode_answer(row) for row in target_rows.tolist()],
        "second_position_eos_fraction": float(np.mean(generated_rows[:, 1] == tokenizer.eos_id)),
    }
    with outputs_path.open("a", encoding="utf-8") as outputs_file:
        outputs_file.write(json.dumps(record, sort_keys=True) + "\n")


def _read_step_metrics(device_metrics: Mapping[str, Array]) -> tuple[dict[str, Any], np.ndarray]:
    """Copy one step's replicated metrics to the host, unsharding generated tokens."""
    assert "generated_token_ids" in device_metrics
    generated = _unshard_token_ids(device_metrics["generated_token_ids"])
    host_metrics = {
        key: np.asarray(jax.device_get(value[0]))
        for key, value in device_metrics.items()
        if key != "generated_token_ids"
    }
    return host_metrics, generated


def _selected_gradient_norm(
    component_gradients: Any,
    parts: tuple[str, ...],
    shared_encoder_reference: Any | None = None,
) -> Array:
    """Measure selected gradients after isolating a tied sender use when needed."""
    selected = {part: component_gradients[part] for part in parts}
    if shared_encoder_reference is not None:
        shared_encoder_part = "shared_sender_receiver_encoder"
        selected[shared_encoder_part] = jax.tree_util.tree_map(
            lambda combined, receiver: combined - receiver,
            component_gradients[shared_encoder_part],
            shared_encoder_reference[shared_encoder_part],
        )
    return optax.global_norm(selected)


def _gate_activation_statistics(gate: Array, valid: Array | None = None) -> Array:
    """Return globally weighted gate mean and near-zero/near-one fractions."""
    if valid is None:
        weights = jnp.ones_like(gate, dtype=jnp.float32)
    else:
        assert valid.shape == gate.shape[:-1]
        weights = jnp.broadcast_to(valid[..., None], gate.shape).astype(jnp.float32)
    local_sums = jnp.stack(
        (
            jnp.sum(gate.astype(jnp.float32) * weights),
            jnp.sum((gate < 0.05).astype(jnp.float32) * weights),
            jnp.sum((gate > 0.95).astype(jnp.float32) * weights),
        )
    )
    global_sums = jax.lax.psum(local_sums, axis_name="devices")
    global_count = jax.lax.psum(jnp.sum(weights), axis_name="devices")
    return global_sums / global_count


def build_parallel_train_step(
    model: JaxCommunicationModel,
    collect_path_gradients: bool = True,
    collect_diagnostics: bool = True,
) -> Any:
    """Compile one data-parallel optimization step across local TPU devices."""
    if collect_path_gradients and not collect_diagnostics:
        raise ValueError("path gradients require diagnostic collection")
    sender_parameter_parts = ("sender", "token_embedding")
    receiver_parameter_parts = ("receiver", "token_embedding")
    if model.config.tie_sender_receiver_encoder:
        shared_encoder_part = "shared_sender_receiver_encoder"
        sender_parameter_parts = (*sender_parameter_parts, shared_encoder_part)
        receiver_parameter_parts = (*receiver_parameter_parts, shared_encoder_part)

    def component_gradients(
        params: Any,
        batch: JaxExampleBatch,
        dropout_key: Array,
        stop_message_1: bool,
        stop_message_2: bool,
        stop_receiver_tokens: bool,
    ) -> Any:
        """Return one stop-gradient slice of the distributed training gradient."""

        def loss_function(current_params: Any) -> Array:
            return model.apply(
                {"params": current_params},
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                batch.answer_token_ids,
                deterministic=False,
                stop_message_1=stop_message_1,
                stop_message_2=stop_message_2,
                stop_receiver_tokens=stop_receiver_tokens,
                rngs={"dropout": dropout_key},
            ).loss

        return jax.lax.pmean(jax.grad(loss_function)(params), axis_name="devices")

    def train_step(
        state: TrainState,
        batch: JaxExampleBatch,
        dropout_key: Array,
    ) -> tuple[TrainState, dict[str, Array]]:
        """Average gradients across devices and apply one optimizer update."""

        def loss_function(params: Any) -> tuple[Array, JaxCommunicationOutput]:
            output = model.apply(
                {"params": params},
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                batch.answer_token_ids,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            return output.loss, output

        (loss, output), gradients = jax.value_and_grad(loss_function, has_aux=True)(state.params)
        gradients = jax.lax.pmean(gradients, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        params_for_path_gradients = state.params
        state = state.apply_gradients(grads=gradients)
        metrics = {
            "loss": loss,
            "gradient_norm": optax.global_norm(gradients),
        }
        if not collect_diagnostics:
            return state, metrics
        generated_token_ids = model.apply(
            {"params": state.params},
            batch.agent_1_token_ids,
            batch.agent_2_token_ids,
            batch.question_token_ids,
            method=model.greedy_answer_token_ids,
        )
        metrics.update({
            "weight_norm": optax.global_norm(state.params),
            "weight_norm_embedding": optax.global_norm(state.params["token_embedding"]),
            "weight_norm_sender": optax.global_norm(state.params["sender"]),
            "weight_norm_receiver": optax.global_norm(state.params["receiver"]),
            "activation_rms_message_1": _global_root_mean_square(output.message_1),
            "activation_rms_message_2": _global_root_mean_square(output.message_2),
            "activation_rms_receiver": _global_root_mean_square(output.receiver_activation),
            "message_1_histogram": jax.lax.psum(
                _activation_histogram_counts(output.message_1), axis_name="devices"
            ),
            "message_2_histogram": jax.lax.psum(
                _activation_histogram_counts(output.message_2), axis_name="devices"
            ),
            "receiver_activation_histogram": jax.lax.psum(
                _activation_histogram_counts(output.receiver_activation), axis_name="devices"
            ),
            "second_position_eos_fraction": jax.lax.pmean(
                jnp.mean((generated_token_ids[:, 1] == model.config.eos_id).astype(jnp.float32)),
                axis_name="devices",
            ),
            "generated_token_ids": generated_token_ids,
        })
        if model.config.tie_sender_receiver_encoder:
            metrics["weight_norm_shared_encoder"] = optax.global_norm(
                state.params["shared_sender_receiver_encoder"]
            )
        if model.config.role_gate_channels is not None:
            assert output.sender_gate_1 is not None
            assert output.sender_gate_2 is not None
            assert output.receiver_gate is not None
            sender_gate_statistics = _gate_activation_statistics(
                jnp.concatenate((output.sender_gate_1, output.sender_gate_2), axis=1)
            )
            receiver_gate_valid = jnp.concatenate(
                (
                    jnp.ones(
                        (batch.question_token_ids.shape[0], model.config.message_length * 2),
                        dtype=jnp.bool_,
                    ),
                    batch.question_token_ids != model.config.padding_id,
                ),
                axis=1,
            )
            receiver_gate_statistics = _gate_activation_statistics(
                output.receiver_gate,
                receiver_gate_valid,
            )
            metrics.update({
                "sender_gate_mean": sender_gate_statistics[0],
                "sender_gate_near_zero_fraction": sender_gate_statistics[1],
                "sender_gate_near_one_fraction": sender_gate_statistics[2],
                "receiver_gate_mean": receiver_gate_statistics[0],
                "receiver_gate_near_zero_fraction": receiver_gate_statistics[1],
                "receiver_gate_near_one_fraction": receiver_gate_statistics[2],
            })
        if collect_path_gradients:
            # These reuse the update's dropout key so each path is a slice of the same noise.
            receiver_gradients = component_gradients(
                params_for_path_gradients,
                batch,
                dropout_key,
                True,
                True,
                False,
            )
            agent_1_gradients = component_gradients(
                params_for_path_gradients,
                batch,
                dropout_key,
                False,
                True,
                True,
            )
            agent_2_gradients = component_gradients(
                params_for_path_gradients,
                batch,
                dropout_key,
                True,
                False,
                True,
            )
            shared_encoder_reference = (
                receiver_gradients if model.config.tie_sender_receiver_encoder else None
            )
            metrics["gradient_norm_agent_1"] = _selected_gradient_norm(
                agent_1_gradients,
                sender_parameter_parts,
                shared_encoder_reference,
            )
            metrics["gradient_norm_agent_2"] = _selected_gradient_norm(
                agent_2_gradients,
                sender_parameter_parts,
                shared_encoder_reference,
            )
            metrics["gradient_norm_receiver"] = _selected_gradient_norm(
                receiver_gradients,
                receiver_parameter_parts,
            )
        return state, metrics

    return jax.pmap(train_step, axis_name="devices")



def build_parallel_evaluation_step(
    model: JaxCommunicationModel,
    collect_message_interventions: bool = False,
    wrong_world_shift: int = 1,
) -> Any:
    """Compile exact-match evaluation across TPU devices."""

    intervention_names = {
        "zero_message": "zero",
        "wrong_world_message": "wrong_world",
        "swapped_sender": "swap_senders",
        "agent_1_only": "agent_1_only",
        "agent_2_only": "agent_2_only",
    }

    def evaluation_step(params: Any, batch: JaxExampleBatch) -> dict[str, Array]:
        """Return globally summed validation statistics."""
        def intervention_statistics(intervention: str) -> tuple[Array, Array]:
            """Return correct-count and joint-log-probability sums for one intervention."""
            output = model.apply(
                {"params": params},
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                batch.answer_token_ids,
                deterministic=True,
                message_intervention=intervention,
                wrong_world_shift=wrong_world_shift,
            )
            generated = model.apply(
                {"params": params},
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                message_intervention=intervention,
                wrong_world_shift=wrong_world_shift,
                method=model.greedy_answer_token_ids,
            )
            correct = jnp.all(generated == batch.answer_token_ids, axis=-1).sum()
            return correct, output.joint_log_probability.sum()

        correct, joint_log_probability = intervention_statistics("intact")
        statistics: dict[str, Array] = {
            "correct": correct,
            "count": jnp.asarray(batch.answer_token_ids.shape[0], dtype=jnp.int32),
            "joint_log_probability": joint_log_probability,
        }
        if collect_message_interventions:
            for metric_prefix, intervention in intervention_names.items():
                intervention_correct, intervention_joint_log_probability = intervention_statistics(
                    intervention
                )
                statistics[f"{metric_prefix}_correct"] = intervention_correct
                statistics[f"{metric_prefix}_joint_log_probability"] = (
                    intervention_joint_log_probability
                )
        return jax.lax.psum(statistics, axis_name="devices")

    return jax.pmap(evaluation_step, axis_name="devices")


def _validated_wrong_world_shift(
    batch_size: int,
    local_device_count: int,
    questions_per_world: int,
) -> int:
    """Return one world-block shift after validating device-local evaluation layout."""
    per_device_batch = batch_size // local_device_count
    if per_device_batch % questions_per_world != 0:
        raise ValueError("message interventions require world-aligned per-device batches")
    if per_device_batch < questions_per_world * 2:
        raise ValueError("message interventions require at least two worlds per device")
    return questions_per_world


def shard_batch(batch: JaxExampleBatch, local_device_count: int) -> JaxExampleBatch:
    """Reshape a global batch into one leading axis per local device."""
    batch_size = batch.answer_token_ids.shape[0]
    if batch_size % local_device_count != 0:
        raise ValueError("global batch size must be divisible by the local TPU device count")
    per_device_batch = batch_size // local_device_count
    return jax.tree_util.tree_map(
        lambda array: array.reshape((local_device_count, per_device_batch, *array.shape[1:])),
        batch,
    )


def evaluate_exact_match(
    params: Any,
    examples: TokenizedExamples,
    batch_size: int,
    local_device_count: int,
    parallel_evaluation_step: Any,
) -> dict[str, float]:
    """Measure greedy exact match and mean joint answer log probability."""
    if examples.size % batch_size != 0:
        raise ValueError("validation example count must be divisible by batch size")
    totals: dict[str, float] = {}
    for start in range(0, examples.size, batch_size):
        batch = examples.select(range(start, start + batch_size))
        statistics = parallel_evaluation_step(params, shard_batch(batch, local_device_count))
        host_statistics = jax.device_get(jax.tree_util.tree_map(lambda value: value[0], statistics))
        for key, value in host_statistics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    count = totals.pop("count")
    results = {
        "exact_match": totals.pop("correct") / count,
        "mean_joint_log_probability": totals.pop("joint_log_probability") / count,
    }
    for key, value in totals.items():
        if key.endswith("_correct"):
            results[f"{key.removesuffix('_correct')}_exact_match"] = value / count
        else:
            metric_prefix = key.removesuffix("_joint_log_probability")
            results[f"{metric_prefix}_mean_joint_log_probability"] = value / count
    return results


def _serializable_config(config: TrainConfig) -> dict[str, Any]:
    """Convert path values in the effective configuration to strings."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _initialize_wandb(config: TrainConfig) -> Any | None:
    """Start W&B on launcher-designated physical worker zero before JAX."""
    if not config.wandb_enabled:
        return None
    worker_index = os.environ.get("ELF_TPU_WORKER_INDEX")
    if config.device == "tpu":
        if worker_index is None or not worker_index.isdigit():
            raise ValueError("ELF_TPU_WORKER_INDEX must identify physical worker zero before JAX starts")
        if int(worker_index) != 0:
            return None
    import wandb

    original_platforms = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        return wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or config.run_id,
            job_type="communication-jax",
            config=_serializable_config(config),
            resume="never",
            settings=wandb.Settings(x_disable_stats=True),
        )
    finally:
        if original_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = original_platforms


def _local_replica(value: Any) -> Any:
    """Read one addressable pmap replica without cross-host array indexing."""
    if not isinstance(value, jax.Array):
        return value
    first_shard = value.addressable_shards[0].data
    if first_shard.shape[0] != 1:
        raise ValueError("replicated pmap array did not have a singleton local shard axis")
    return np.asarray(jax.device_get(jnp.squeeze(first_shard, axis=0)))


def save_checkpoint(state: TrainState, step: int, output_path: Path) -> None:
    """Atomically save one process-local replica of the final training state."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {output_path}")
    local_state = jax.tree_util.tree_map(_local_replica, state)
    payload = serialization.to_bytes({"step": step, "state": local_state})
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"refusing to overwrite partial checkpoint {temporary_path}")
    temporary_path.write_bytes(payload)
    temporary_path.replace(output_path)


def _append_metrics(metrics_path: Path | None, metrics: Mapping[str, Any]) -> None:
    """Append one JSON metric record when an output directory is configured."""
    if metrics_path is None:
        return
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(dict(metrics), sort_keys=True) + "\n")


def train(config: TrainConfig) -> dict[str, Any]:
    """Train the Flax communication model and return final evaluation metrics."""
    config.validate()
    output_directory = config.output_path.parent if config.output_path is not None else None
    if output_directory is not None:
        if output_directory.exists():
            raise FileExistsError(f"refusing to reuse run output directory {output_directory}")
        output_directory.mkdir(parents=False)
        (output_directory / "config.yml").write_text(
            yaml.safe_dump(_serializable_config(config), sort_keys=True), encoding="utf-8"
        )
    wandb_run = _initialize_wandb(config)
    try:
        if config.device == "tpu":
            jax.distributed.initialize()
        devices = jax.devices()
        if not devices or devices[0].platform != config.device:
            found_platforms = [device.platform for device in devices]
            raise RuntimeError(f"requested {config.device} backend, found {found_platforms}")
        if config.device == "tpu" and jax.process_count() != 1:
            raise RuntimeError("this optimized conlang runner currently requires a single-host TPU")
        local_device_count = jax.local_device_count()
        if config.batch_size % local_device_count != 0:
            raise ValueError("batch size must be divisible by the local device count")
        dataset_directory = resolve_training_dataset(
            config.dataset_dir,
            config.dataset_prefix,
            cache_dir=config.dataset_cache_dir,
        )
        verified_dataset = verify_dataset(dataset_directory)
        wrong_world_shift = 1
        if config.evaluation_message_interventions:
            wrong_world_shift = _validated_wrong_world_shift(
                config.batch_size,
                local_device_count,
                verified_dataset["questions_per_world"],
            )
        train_examples = load_examples(resolve_split_path(dataset_directory, "train"), config.limit_examples)
        validation_examples = load_examples(
            resolve_split_path(dataset_directory, "validation"), config.eval_examples
        )
        tokenizer = build_task_tokenizer()
        tokenized_train = tokenize_examples(train_examples, tokenizer, config)
        tokenized_validation = tokenize_examples(validation_examples, tokenizer, config)
        batch_iterator = iter_epoch_batches(
            tokenized_train,
            config.batch_size,
            random.Random(config.seed),
            repeat_examples_to_batch_size=config.repeat_training_examples_to_batch_size,
        )
        first_batch = next(batch_iterator)
        model = JaxCommunicationModel(_model_config(config, tokenizer))
        initialization_optimizer = optax.identity()
        initial_state = _initialize_state(model, initialization_optimizer, first_batch, config.seed)
        optimizer = _optimizer_for_params(config, initial_state.params)
        state = TrainState.create(apply_fn=model.apply, params=initial_state.params, tx=optimizer)
        replicated_state = jax_utils.replicate(state)
        parallel_train_step = build_parallel_train_step(
            model,
            collect_path_gradients=False,
            collect_diagnostics=False,
        )
        parallel_logged_train_step = build_parallel_train_step(model, collect_path_gradients=True)
        parallel_evaluation_step = build_parallel_evaluation_step(
            model,
            collect_message_interventions=config.evaluation_message_interventions,
            wrong_world_shift=wrong_world_shift,
        )
        metrics_path = None if output_directory is None else output_directory / "metrics.jsonl"
        outputs_path = None if output_directory is None else output_directory / "outputs.jsonl"
        print(
            f"run_id={config.run_id} backend={devices[0].platform} devices={jax.device_count()} "
            f"dataset={dataset_directory} verified_questions={verified_dataset['questions']} "
            f"train_examples={tokenized_train.size} validation_examples={tokenized_validation.size} "
            f"batch_size={config.batch_size} precision={config.precision}"
        )
        interval_started_at = time.perf_counter()
        interval_start_step = 0
        last_metrics: dict[str, Any] = {"step": 0.0, "train_loss": 0.0}
        base_dropout_key = jax.random.PRNGKey(config.seed + 1)
        for step in range(1, config.steps + 1):
            batch = first_batch if step == 1 else next(batch_iterator)
            sharded_batch = shard_batch(batch, local_device_count)
            dropout_keys = jax.random.split(jax.random.fold_in(base_dropout_key, step), local_device_count)
            should_log_scalars = step == config.throughput_warmup_steps or (
                step > config.throughput_warmup_steps and (step % config.log_every == 0 or step == config.steps)
            )
            step_function = parallel_logged_train_step if should_log_scalars else parallel_train_step
            replicated_state, device_metrics = step_function(
                replicated_state, sharded_batch, dropout_keys
            )
            should_evaluate = step % config.eval_every == 0 or step == config.steps
            host_metrics = None
            generated_token_ids = None
            if should_log_scalars:
                host_metrics, generated_token_ids = _read_step_metrics(device_metrics)
                if outputs_path is not None:
                    _append_model_outputs(
                        outputs_path,
                        step,
                        generated_token_ids,
                        batch.answer_token_ids,
                        tokenizer,
                    )
            pending_wandb: dict[str, Any] | None = None
            reset_interval_timer = False
            if step == config.throughput_warmup_steps and host_metrics is not None:
                warmup_record = _host_training_record(
                    step,
                    host_metrics,
                    None,
                    generated_token_ids,
                )
                print(
                    f"step={step} train_loss={warmup_record['train_loss']:.4f} "
                    f"gradient_norm={warmup_record['gradient_norm']:.4f} throughput=warmup"
                )
                _append_metrics(metrics_path, warmup_record)
                pending_wandb = warmup_record
                reset_interval_timer = True
            if (
                step > config.throughput_warmup_steps
                and (step % config.log_every == 0 or step == config.steps)
                and host_metrics is not None
            ):
                elapsed = time.perf_counter() - interval_started_at
                interval_steps = step - interval_start_step
                examples_per_second = interval_steps * config.batch_size / elapsed
                last_metrics = _host_training_record(
                    step,
                    host_metrics,
                    examples_per_second,
                    generated_token_ids,
                )
                print(
                    f"step={step} train_loss={last_metrics['train_loss']:.4f} "
                    f"gradient_norm={last_metrics['gradient_norm']:.4f} "
                    f"examples_per_second={examples_per_second:.2f}"
                )
                _append_metrics(metrics_path, last_metrics)
                pending_wandb = last_metrics
                reset_interval_timer = True
            if should_evaluate:
                evaluation = evaluate_exact_match(
                    replicated_state.params,
                    tokenized_validation,
                    config.batch_size,
                    local_device_count,
                    parallel_evaluation_step,
                )
                last_metrics.update(evaluation)
                print(
                    f"step={step} exact_match={evaluation['exact_match']:.4f} "
                    f"mean_joint_log_probability={evaluation['mean_joint_log_probability']:.4f}"
                )
                evaluation_record = {"step": float(step), **evaluation}
                _append_metrics(metrics_path, evaluation_record)
                eval_payload = {f"eval/{key}": value for key, value in evaluation.items()}
                pending_wandb = eval_payload if pending_wandb is None else {**pending_wandb, **eval_payload}
                reset_interval_timer = True
            if pending_wandb is not None and wandb_run is not None:
                wandb_run.log(_as_wandb_log(pending_wandb), step=step)
            if reset_interval_timer:
                interval_started_at = time.perf_counter()
                interval_start_step = step
        if config.output_path is not None:
            save_checkpoint(replicated_state, config.steps, config.output_path)
        print(f"TRAINING_COMPLETE run_id={config.run_id} step={config.steps}")
        if wandb_run is not None:
            wandb_run.finish(exit_code=0)
        return last_metrics
    except BaseException:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise


def _load_config_values(path: Path) -> dict[str, Any]:
    """Load and validate one YAML mapping against TrainConfig fields."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("training config must contain one YAML mapping")
    allowed_fields = {field.name for field in fields(TrainConfig)}
    unknown_fields = sorted(set(loaded) - allowed_fields)
    if unknown_fields:
        raise ValueError(f"unknown training config fields: {', '.join(unknown_fields)}")
    values = dict(loaded)
    for path_field in ("dataset_dir", "dataset_cache_dir", "output_path"):
        if values.get(path_field) is not None:
            values[path_field] = Path(values[path_field])
    return values


def parse_arguments(arguments: Sequence[str] | None = None) -> TrainConfig:
    """Parse a primary YAML config plus narrow command-line overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-gradient-norm", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-examples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "tpu"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--disable-wandb", action="store_true")
    parsed = parser.parse_args(arguments)
    values = {} if parsed.config is None else _load_config_values(parsed.config)
    command_line_values = {
        "dataset_dir": parsed.dataset_dir,
        "dataset_cache_dir": parsed.dataset_cache_dir,
        "batch_size": parsed.batch_size,
        "learning_rate": parsed.learning_rate,
        "weight_decay": parsed.weight_decay,
        "max_gradient_norm": parsed.max_gradient_norm,
        "steps": parsed.steps,
        "eval_every": parsed.eval_every,
        "eval_examples": parsed.eval_examples,
        "seed": parsed.seed,
        "device": parsed.device,
        "output_path": parsed.output,
        "limit_examples": parsed.limit_examples,
        "wandb_project": parsed.wandb_project,
    }
    values.update({key: value for key, value in command_line_values.items() if value is not None})
    if parsed.disable_wandb:
        values["wandb_enabled"] = False
    config = TrainConfig(**values)
    config.validate()
    return config


def main(arguments: Sequence[str] | None = None) -> int:
    """Train the JAX communication model from the command line."""
    train(parse_arguments(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

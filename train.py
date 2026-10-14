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

from jax_models import JaxCommunicationModel, JaxCommunicationModelConfig
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
    wandb_project: str = "conlangs"
    wandb_run_name: str | None = None
    wandb_enabled: bool = True
    hidden_size: int = 128
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
) -> Iterator[JaxExampleBatch]:
    """Yield shuffled, full, fixed-shape batches indefinitely."""
    if examples.size < batch_size:
        raise ValueError("training dataset is smaller than one full batch")
    while True:
        order = list(range(examples.size))
        random_generator.shuffle(order)
        for start in range(0, examples.size - batch_size + 1, batch_size):
            yield examples.select(order[start : start + batch_size])


def _model_config(config: TrainConfig, tokenizer: TextTokenizer) -> JaxCommunicationModelConfig:
    """Build the Flax model configuration from the effective run settings."""
    dtype = jnp.bfloat16 if config.precision == "bfloat16" else jnp.float32
    return JaxCommunicationModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_size,
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


def build_parallel_train_step(model: JaxCommunicationModel) -> Any:
    """Compile one data-parallel optimization step across local TPU devices."""

    def train_step(
        state: TrainState,
        batch: JaxExampleBatch,
        dropout_key: Array,
    ) -> tuple[TrainState, dict[str, Array]]:
        """Average gradients across devices and apply one optimizer update."""

        def loss_function(params: Any) -> Array:
            output = model.apply(
                {"params": params},
                batch.agent_1_token_ids,
                batch.agent_2_token_ids,
                batch.question_token_ids,
                batch.answer_token_ids,
                deterministic=False,
                rngs={"dropout": dropout_key},
            )
            return output.loss

        loss, gradients = jax.value_and_grad(loss_function)(state.params)
        gradients = jax.lax.pmean(gradients, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        gradient_norm = optax.global_norm(gradients)
        state = state.apply_gradients(grads=gradients)
        return state, {"loss": loss, "gradient_norm": gradient_norm}

    return jax.pmap(train_step, axis_name="devices")


def build_parallel_evaluation_step(model: JaxCommunicationModel) -> Any:
    """Compile exact-match evaluation across TPU devices."""

    def evaluation_step(params: Any, batch: JaxExampleBatch) -> dict[str, Array]:
        """Return globally summed validation statistics."""
        output = model.apply(
            {"params": params},
            batch.agent_1_token_ids,
            batch.agent_2_token_ids,
            batch.question_token_ids,
            batch.answer_token_ids,
            deterministic=True,
        )
        generated = model.apply(
            {"params": params},
            batch.agent_1_token_ids,
            batch.agent_2_token_ids,
            batch.question_token_ids,
            method=model.greedy_answer_token_ids,
        )
        correct = jnp.all(generated == batch.answer_token_ids, axis=-1).sum()
        statistics = {
            "correct": correct,
            "count": jnp.asarray(batch.answer_token_ids.shape[0], dtype=jnp.int32),
            "joint_log_probability": output.joint_log_probability.sum(),
        }
        return jax.lax.psum(statistics, axis_name="devices")

    return jax.pmap(evaluation_step, axis_name="devices")


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
    correct = 0
    count = 0
    joint_log_probability = 0.0
    for start in range(0, examples.size, batch_size):
        batch = examples.select(range(start, start + batch_size))
        statistics = parallel_evaluation_step(params, shard_batch(batch, local_device_count))
        host_statistics = jax.device_get(jax.tree_util.tree_map(lambda value: value[0], statistics))
        correct += int(host_statistics["correct"])
        count += int(host_statistics["count"])
        joint_log_probability += float(host_statistics["joint_log_probability"])
    return {
        "exact_match": correct / count,
        "mean_joint_log_probability": joint_log_probability / count,
    }


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


def _append_metrics(metrics_path: Path | None, metrics: Mapping[str, float]) -> None:
    """Append one JSON metric record when an output directory is configured."""
    if metrics_path is None:
        return
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(dict(metrics), sort_keys=True) + "\n")


def train(config: TrainConfig) -> dict[str, float]:
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
        train_examples = load_examples(resolve_split_path(dataset_directory, "train"), config.limit_examples)
        validation_examples = load_examples(
            resolve_split_path(dataset_directory, "validation"), config.eval_examples
        )
        tokenizer = build_task_tokenizer()
        tokenized_train = tokenize_examples(train_examples, tokenizer, config)
        tokenized_validation = tokenize_examples(validation_examples, tokenizer, config)
        batch_iterator = iter_epoch_batches(tokenized_train, config.batch_size, random.Random(config.seed))
        first_batch = next(batch_iterator)
        model = JaxCommunicationModel(_model_config(config, tokenizer))
        initialization_optimizer = optax.identity()
        initial_state = _initialize_state(model, initialization_optimizer, first_batch, config.seed)
        optimizer = _optimizer_for_params(config, initial_state.params)
        state = TrainState.create(apply_fn=model.apply, params=initial_state.params, tx=optimizer)
        replicated_state = jax_utils.replicate(state)
        parallel_train_step = build_parallel_train_step(model)
        parallel_evaluation_step = build_parallel_evaluation_step(model)
        metrics_path = None if output_directory is None else output_directory / "metrics.jsonl"
        print(
            f"run_id={config.run_id} backend={devices[0].platform} devices={jax.device_count()} "
            f"dataset={dataset_directory} verified_questions={verified_dataset['questions']} "
            f"train_examples={tokenized_train.size} validation_examples={tokenized_validation.size} "
            f"batch_size={config.batch_size} precision={config.precision}"
        )
        interval_started_at = time.perf_counter()
        interval_start_step = 0
        last_metrics: dict[str, float] = {"step": 0.0, "train_loss": 0.0}
        base_dropout_key = jax.random.PRNGKey(config.seed + 1)
        for step in range(1, config.steps + 1):
            batch = first_batch if step == 1 else next(batch_iterator)
            sharded_batch = shard_batch(batch, local_device_count)
            dropout_keys = jax.random.split(jax.random.fold_in(base_dropout_key, step), local_device_count)
            replicated_state, device_metrics = parallel_train_step(
                replicated_state, sharded_batch, dropout_keys
            )
            if step == config.throughput_warmup_steps:
                host_metrics = jax.device_get(
                    jax.tree_util.tree_map(lambda value: value[0], device_metrics)
                )
                warmup_record = {
                    "step": float(step),
                    "train_loss": float(host_metrics["loss"]),
                    "gradient_norm": float(host_metrics["gradient_norm"]),
                }
                print(
                    f"step={step} train_loss={warmup_record['train_loss']:.4f} "
                    f"gradient_norm={warmup_record['gradient_norm']:.4f} throughput=warmup"
                )
                _append_metrics(metrics_path, warmup_record)
                if wandb_run is not None:
                    wandb_run.log(warmup_record, step=step)
                interval_started_at = time.perf_counter()
                interval_start_step = step
            if step > config.throughput_warmup_steps and (
                step % config.log_every == 0 or step == config.steps
            ):
                host_metrics = jax.device_get(
                    jax.tree_util.tree_map(lambda value: value[0], device_metrics)
                )
                elapsed = time.perf_counter() - interval_started_at
                interval_steps = step - interval_start_step
                examples_per_second = interval_steps * config.batch_size / elapsed
                last_metrics = {
                    "step": float(step),
                    "train_loss": float(host_metrics["loss"]),
                    "gradient_norm": float(host_metrics["gradient_norm"]),
                    "examples_per_second": examples_per_second,
                }
                print(
                    f"step={step} train_loss={last_metrics['train_loss']:.4f} "
                    f"gradient_norm={last_metrics['gradient_norm']:.4f} "
                    f"examples_per_second={examples_per_second:.2f}"
                )
                _append_metrics(metrics_path, last_metrics)
                if wandb_run is not None:
                    wandb_run.log(last_metrics, step=step)
                interval_started_at = time.perf_counter()
                interval_start_step = step
            if step % config.eval_every == 0 or step == config.steps:
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
                if wandb_run is not None:
                    wandb_run.log({f"eval/{key}": value for key, value in evaluation.items()}, step=step)
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

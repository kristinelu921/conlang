"""Run a bounded paired causal-language-model ELR experiment on one TPU host."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import flax.linen as nn
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from elr_probe.dynamics import controlled_update, matrix_diagnostics
from elr_probe.tracking import tracking_metrics


class LanguageModel(nn.Module):
    """Small pre-norm decoder with learned RMS gains, QK norm, RoPE, and SwiGLU."""

    width: int
    layers: int
    heads: int
    vocabulary_size: int

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        embedding = nn.Embed(self.vocabulary_size, self.width, name="embedding")
        hidden = embedding(tokens)
        head_width = self.width // self.heads
        positions = jnp.arange(tokens.shape[1])
        frequencies = 10000.0 ** (-jnp.arange(0, head_width, 2) / head_width)
        phases = positions[:, None] * frequencies[None, :]
        cosine = jnp.cos(phases)[None, :, None, :]
        sine = jnp.sin(phases)[None, :, None, :]
        causal_mask = positions[:, None] >= positions[None, :]
        for layer in range(self.layers):
            normalized = nn.RMSNorm(name=f"layer_{layer}_attention_norm")(hidden)
            projections = []
            for kind in ("q", "k", "v"):
                projected = nn.Dense(self.width, use_bias=False, name=f"layer_{layer}_{kind}")(normalized)
                projected = projected.reshape(*tokens.shape, self.heads, head_width)
                if kind != "v":
                    projected = nn.RMSNorm(name=f"layer_{layer}_{kind}_norm")(projected)
                    even, odd = projected[..., ::2], projected[..., 1::2]
                    projected = jnp.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1).reshape(projected.shape)
                projections.append(projected)
            query, key, value = projections
            scores = jnp.einsum("bthd,bshd->bhts", query, key) / head_width**0.5
            attention = jax.nn.softmax(jnp.where(causal_mask, scores, -1e30), axis=-1)
            attended = jnp.einsum("bhts,bshd->bthd", attention, value).reshape(*tokens.shape, self.width)
            hidden = hidden + nn.Dense(self.width, use_bias=False, name=f"layer_{layer}_o")(attended)
            normalized = nn.RMSNorm(name=f"layer_{layer}_mlp_norm")(hidden)
            gate = nn.Dense(4 * self.width, use_bias=False, name=f"layer_{layer}_gate")(normalized)
            up = nn.Dense(4 * self.width, use_bias=False, name=f"layer_{layer}_up")(normalized)
            hidden = hidden + nn.Dense(self.width, use_bias=False, name=f"layer_{layer}_down")(jax.nn.silu(gate) * up)
        return embedding.attend(nn.RMSNorm(name="final_norm")(hidden))


def make_step(model: LanguageModel, optimizer: optax.GradientTransformation) -> Any:
    """Compile a data-parallel update with per-matrix diagnostics before projection."""
    @functools.partial(jax.pmap, axis_name="data", in_axes=(0, 0, 0, None, None, None, None))
    def step(
        parameters: dict[str, Any], optimizer_state: Any, batch: jax.Array,
        learning_rate: jax.Array, next_norm_multiplier: jax.Array,
        target_norms: dict[tuple[str, ...], jax.Array], controlled: bool,
    ) -> tuple[dict[str, Any], Any, dict[str, jax.Array]]:
        def loss_function(weights: dict[str, Any]) -> jax.Array:
            logits = model.apply({"params": weights}, batch[:, :-1])
            return optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:]).mean()

        loss, gradients = jax.value_and_grad(loss_function)(parameters)
        gradients = jax.lax.pmean(gradients, "data")
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
        flat_parameters = flatten_dict(parameters)
        flat_updates = flatten_dict(updates)
        next_parameters = {}
        metrics = {"loss": jax.lax.pmean(loss, "data")}
        for path, parameter in flat_parameters.items():
            provisional = parameter + learning_rate * flat_updates[path]
            if parameter.ndim == 2:
                diagnostics = matrix_diagnostics(parameter, provisional)
                diagnostics["elr"] = learning_rate / diagnostics["norm"]
                diagnostics["unscaled_update_norm"] = jnp.linalg.norm(flat_updates[path])
                for name, value in diagnostics.items():
                    metrics["/".join(path) + "/" + name] = value
                projected = controlled_update(parameter, flat_updates[path], learning_rate, target_norms[path] * next_norm_multiplier)
                provisional = jnp.where(controlled, projected, provisional)
            next_parameters[path] = provisional
        return unflatten_dict(next_parameters), next_optimizer_state, metrics

    return step


def run(config: dict[str, Any], data_path: Path, output_path: Path, tracking_run: Any) -> None:
    """Train paired arms with identical minibatches and an identical warmup state."""
    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise RuntimeError("This pilot requires an authorized single-host TPU")
    if not 0 < config["warmup_steps"] < config["steps"]:
        raise ValueError("Warmup must be positive and shorter than training")
    if min(config[name] for name in ("batch_size", "sequence_length", "checkpoint_every", "width", "layers", "heads", "learning_rate", "final_norm_multiplier")) <= 0:
        raise ValueError("Model, training, and norm schedule dimensions must be positive")
    if config["width"] % (2 * config["heads"]):
        raise ValueError("RoPE requires an even integer head width")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Specify distinct paired seeds")
    if output_path.exists():
        raise FileExistsError(output_path)
    if not str(data_path.resolve()).startswith("/home/") or "/elf-data/" not in str(data_path.resolve()):
        raise ValueError("Dataset must be verified node-local under /home/<user>/elf-data/")
    if not str(output_path.resolve()).startswith("/home/") or "/elf-runs/" not in str(output_path.resolve()):
        raise ValueError("Outputs must be node-local under /home/<user>/elf-runs/")
    devices = jax.local_devices()
    if config["batch_size"] % len(devices):
        raise ValueError("Batch size must divide evenly across devices")
    tokens = np.load(data_path, mmap_mode="r")
    if tokens.ndim != 1 or tokens.dtype != np.uint16 or len(tokens) <= config["sequence_length"] + 1:
        raise ValueError("Expected a nonempty one-dimensional uint16 token corpus")
    if int(tokens.max()) >= config["vocabulary_size"]:
        raise ValueError("Corpus token IDs exceed vocabulary")
    output_path.mkdir(parents=True)
    (output_path / "config.yaml").write_text(yaml.safe_dump(config))
    with data_path.open("rb") as source:
        (output_path / "data-sha256.txt").write_text(hashlib.file_digest(source, "sha256").hexdigest())
    model = LanguageModel(config["width"], config["layers"], config["heads"], config["vocabulary_size"])
    optimizer = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8), optax.scale(-1.0))
    step_function = make_step(model, optimizer)
    started = time.monotonic()
    for seed_index, seed in enumerate(config["seeds"]):
        generator = np.random.default_rng(seed)
        parameters = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, config["sequence_length"]), dtype=jnp.int32))["params"]
        assert isinstance(parameters, dict)
        optimizer_state = optimizer.init(parameters)
        targets = {path: jnp.linalg.norm(value) for path, value in flatten_dict(parameters).items() if value.ndim == 2}
        replicate = functools.partial(jax.device_put_replicated, devices=devices)
        states = {"baseline": (replicate(parameters), replicate(optimizer_state))}
        loss_averages: dict[str, float] = {}
        with (output_path / f"metrics-{seed}.jsonl").open("x", buffering=1) as metrics_file:
            for step_index in range(config["steps"]):
                offsets = generator.integers(0, len(tokens) - config["sequence_length"], size=config["batch_size"])
                batch = np.stack([tokens[offset:offset + config["sequence_length"] + 1] for offset in offsets]).astype(np.int32)
                batch = batch.reshape(len(devices), -1, config["sequence_length"] + 1)
                warmup_steps = config["warmup_steps"]
                controlled = step_index >= warmup_steps
                if step_index == warmup_steps:
                    states["matched"] = states["baseline"]
                    warm_parameters = jax.tree.map(lambda value: value[0], states["baseline"][0])
                    targets = {path: jnp.linalg.norm(value) for path, value in flatten_dict(warm_parameters).items() if value.ndim == 2}
                    loss_averages["matched"] = loss_averages["baseline"]
                record: dict[str, Any] = {"step": step_index, "seed": seed, "controlled": controlled, "elapsed_seconds": time.monotonic() - started}
                for arm in states:
                    fraction = max(0, step_index - warmup_steps) / (config["steps"] - warmup_steps)
                    next_fraction = max(0, step_index + 1 - warmup_steps) / (config["steps"] - warmup_steps)
                    multiplier = 1 + fraction * (config["final_norm_multiplier"] - 1) if arm == "matched" else 1.0
                    next_multiplier = 1 + next_fraction * (config["final_norm_multiplier"] - 1) if arm == "matched" else 1.0
                    learning_rate = config["learning_rate"] * min(1, (step_index + 1) / warmup_steps) * multiplier
                    next_parameters, next_state, metrics = step_function(*states[arm], batch, learning_rate, next_multiplier, targets, controlled)
                    states[arm] = (next_parameters, next_state)
                    observed = {name: float(value[0]) for name, value in jax.device_get(metrics).items()}
                    loss_averages[arm] = 0.99 * loss_averages.get(arm, observed["loss"]) + 0.01 * observed["loss"]
                    observed["loss_ema"] = loss_averages[arm]
                    observed["learning_rate"] = learning_rate
                    record[arm] = observed
                if controlled:
                    comparison = {}
                    for name in record["baseline"]:
                        if name.endswith("/elr"):
                            relative_error = abs(record["baseline"][name] / record["matched"][name] - 1)
                            if relative_error > 2e-5:
                                raise RuntimeError(f"ELR mismatch at step {step_index}: {name}={relative_error}")
                            comparison[name + "_relative_error"] = relative_error
                        if name.endswith("/angle_radians"):
                            denominator = record["matched"][name]
                            comparison[name + "_baseline_over_matched"] = record["baseline"][name] / denominator if denominator > 1e-8 else None
                    comparison["loss_ema_matched_minus_baseline"] = loss_averages["matched"] - loss_averages["baseline"]
                    record["comparison"] = comparison
                metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
                tracking_run.log(tracking_metrics(record), step=seed_index * config["steps"] + step_index)
                if step_index % 20 == 0:
                    print(json.dumps({"seed": seed, "step": step_index, "loss": {arm: record[arm]["loss"] for arm in states}}), flush=True)
                if (step_index + 1) % config["checkpoint_every"] == 0 or step_index + 1 == config["steps"]:
                    checkpoint = {"states": jax.device_get(jax.tree.map(lambda value: value[0], states)), "step": step_index + 1, "seed": seed, "generator": json.dumps(generator.bit_generator.state), "loss_averages": loss_averages, "targets": {"/".join(path): float(value) for path, value in targets.items()}}
                    destination = output_path / f"checkpoint-{seed}-{step_index + 1}.msgpack"
                    temporary = destination.with_suffix(".partial")
                    temporary.write_bytes(serialization.to_bytes(checkpoint))
                    temporary.rename(destination)
                    destination.with_suffix(".sha256").write_text(hashlib.sha256(destination.read_bytes()).hexdigest())
    (output_path / "COMPLETED").write_text(json.dumps({"seeds": config["seeds"], "steps": config["steps"]}))


def main() -> None:
    """Load the explicit pilot configuration and node-local paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text())
    assert isinstance(config, dict)
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    import wandb

    previous_platform = os.environ.get("JAX_PLATFORMS", "tpu")
    os.environ["JAX_PLATFORMS"] = "cpu"
    tracking_directory = arguments.output.parent / "wandb"
    tracking_directory.mkdir(parents=True, exist_ok=True)
    try:
        tracking_run = wandb.init(
            project=config["wandb_project"], name=config["run_id"],
            group="elr-angular-pilot", config=config, resume="never",
            dir=str(tracking_directory), settings=wandb.Settings(x_disable_stats=True),
        )
    finally:
        os.environ["JAX_PLATFORMS"] = previous_platform
    print(f"WANDB_URL={tracking_run.url}", flush=True)
    exit_code = 1
    try:
        for seed in config["seeds"]:
            tracking_run.define_metric(f"seed_{seed}/training_step")
            tracking_run.define_metric(f"seed_{seed}/*", step_metric=f"seed_{seed}/training_step")
        run(config, arguments.data, arguments.output, tracking_run)
        exit_code = 0
    finally:
        tracking_run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()

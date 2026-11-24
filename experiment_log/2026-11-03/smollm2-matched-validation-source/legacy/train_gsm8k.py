"""Fine-tune the verified SmolLM2 checkpoint on GSM8K answers using remote TPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from datasets import load_dataset
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model

from flax_tpu_lm_eval import FlaxTPULanguageModel


def encode_example(
    question: str, answer: str, tokenizer: PreTrainedTokenizerBase, maximum_length: int
) -> tuple[list[int], list[int]]:
    """Tokenize the prompt and solution while supervising only answer tokens and EOS."""
    prompt = "Question: " + question + "\nAnswer:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    complete_tokens = tokenizer.encode(prompt + " " + answer, add_special_tokens=False)
    if complete_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError("Tokenization changes across the prompt/answer boundary.")
    input_tokens = complete_tokens + [tokenizer.eos_token_id]
    if len(input_tokens) > maximum_length:
        raise ValueError("Training example exceeds sequence length; truncation is disabled.")
    loss_mask = [0] * len(prompt_tokens) + [1] * (len(input_tokens) - len(prompt_tokens))
    return input_tokens, loss_mask


def make_batch(
    examples: list[tuple[list[int], list[int]]], indices: np.ndarray,
    batch_size: int, sequence_length: int, padding_token: int,
) -> dict[str, np.ndarray]:
    """Right-pad examples and zero-weight synthetic entries in the final batch."""
    input_ids = np.full((batch_size, sequence_length), padding_token, dtype=np.int32)
    attention_mask = np.zeros_like(input_ids)
    loss_mask = np.zeros_like(input_ids, dtype=np.float32)
    for batch_index, example_index in enumerate(indices):
        token_ids, answer_mask = examples[int(example_index)]
        input_ids[batch_index, : len(token_ids)] = token_ids
        attention_mask[batch_index, : len(token_ids)] = 1
        loss_mask[batch_index, : len(token_ids)] = answer_mask
    # A padded row still needs one visible token for well-defined attention.
    attention_mask[len(indices) :, 0] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "loss_mask": loss_mask}


def tied_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Connect the output projection to the same trainable input embedding."""
    return {"model": parameters["model"],
            "lm_head": {"kernel": parameters["model"]["embed_tokens"]["embedding"].T}}


def masked_loss_sum(
    logits: jax.Array, input_ids: jax.Array, loss_mask: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute shifted answer-token cross-entropy sums and supervised token counts."""
    token_losses = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], input_ids[:, 1:])
    shifted_mask = loss_mask[:, 1:]
    return jnp.sum(token_losses * shifted_mask), jnp.sum(shifted_mask)


def file_record(path: Path) -> dict[str, Any]:
    """Hash a finalized checkpoint artifact without loading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def mirror_directory(directory: Path, bucket_prefix: str) -> None:
    """Upload checkpoint contents before publishing their completion manifest."""
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "complete.json")
    destination = bucket_prefix + "/" + directory.name + "/"
    subprocess.run(["gcloud", "storage", "cp", *map(str, files), destination], check=True)
    subprocess.run(["gcloud", "storage", "cp", str(directory / "complete.json"), destination], check=True)
    print(json.dumps({"event": "artifact_mirrored", "destination": destination}), flush=True)


def save_checkpoint(
    state: TrainState, output_directory: Path, epoch: int,
    configuration: dict[str, Any], source_digest: str,
) -> TrainState:
    """Persist a full optimizer checkpoint and mirror it to the same-region bucket."""
    host_state = jax.tree.map(lambda value: np.asarray(value[0]), state)
    checkpoint_directory = output_directory / f"checkpoint-{int(host_state.step):06d}"
    checkpoint_directory.mkdir()
    state_path = checkpoint_directory / "training_state.msgpack"
    with state_path.open("xb") as handle:
        handle.write(serialization.to_bytes(host_state))
        handle.flush()
        os.fsync(handle.fileno())
    metadata = {"completed_epochs": epoch, "step": int(host_state.step),
                "configuration": configuration, "source_digest": source_digest}
    (checkpoint_directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {"status": "complete", "step": int(host_state.step), "completed_epochs": epoch,
                "files": [file_record(state_path), file_record(checkpoint_directory / "metadata.json")]}
    (checkpoint_directory / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    mirror_directory(checkpoint_directory, configuration["artifact_bucket_prefix"])
    return host_state


def main() -> None:
    """Train all weights with tied embeddings and deterministic, token-weighted updates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text())
    assert isinstance(configuration, dict)
    output_directory = Path(configuration["output_directory"])
    if not str(output_directory).startswith("/home/") or "smollm2-gsm8k/" not in str(output_directory):
        raise ValueError("Only the authorized remote node-local output root is allowed.")
    output_directory.mkdir(exist_ok=False)
    (output_directory / "effective_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    source_digest = hashlib.sha256(Path("source.sha256").read_bytes()).hexdigest()
    backend = FlaxTPULanguageModel(configuration["model"], configuration["model_revision"], batch_size=1)
    if len(backend.devices) != configuration["device_count"]:
        raise ValueError("Unexpected number of TPU devices.")
    del backend.replicated_parameters
    dataset = load_dataset(configuration["dataset"], configuration["dataset_configuration"],
                           revision=configuration["dataset_revision"], split=configuration["training_split"])
    if len(dataset) != configuration["training_examples"]:
        raise ValueError("Unexpected training split size.")
    examples = [encode_example(row["question"], row["answer"], backend.tokenizer,
                               configuration["maximum_sequence_length"]) for row in dataset]
    batch_size = configuration["batch_size_per_device"] * len(backend.devices)
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = steps_per_epoch * configuration["epochs"]
    warmup_steps = max(1, math.ceil(total_steps * configuration["warmup_fraction"]))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=configuration["learning_rate"], warmup_steps=warmup_steps,
        decay_steps=total_steps, end_value=configuration["final_learning_rate"],
    )
    optimizer = optax.chain(optax.clip_by_global_norm(configuration["gradient_clip_norm"]),
                           optax.adamw(schedule, b1=configuration["adam_beta1"],
                                       b2=configuration["adam_beta2"], eps=configuration["adam_epsilon"],
                                       weight_decay=configuration["weight_decay"]))
    with jax.default_device(jax.devices("cpu")[0]):
        host_state = TrainState.create(apply_fn=backend.model.__call__,
                                      params={"model": backend.parameters["model"]}, tx=optimizer)
    start_epoch = 0
    if arguments.resume is not None:
        metadata = json.loads((arguments.resume / "metadata.json").read_text())
        manifest = json.loads((arguments.resume / "complete.json").read_text())
        if file_record(arguments.resume / "training_state.msgpack") != manifest["files"][0]:
            raise ValueError("Resume checkpoint failed its checksum.")
        for key in configuration:
            if key not in {"run_id", "output_directory", "artifact_bucket_prefix"}:
                if configuration[key] != metadata["configuration"][key]:
                    raise ValueError(f"Resume configuration differs: {key}")
        host_state = serialization.from_bytes(host_state, (arguments.resume / "training_state.msgpack").read_bytes())
        start_epoch = metadata["completed_epochs"]
    state = jax.device_put_replicated(host_state, backend.devices)
    del host_state

    def training_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Sum replica gradients with global answer-token weighting and one tied embedding."""
        global_token_count = jax.lax.psum(jnp.sum(batch["loss_mask"][:, 1:]), "devices")

        def objective(parameters: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
            """Keep the head tied inside differentiation so both gradient paths accumulate."""
            logits = backend.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                   params=tied_parameters(parameters), train=True,
                                   dropout_rng=jax.random.fold_in(jax.random.PRNGKey(configuration["seed"]), state.step)).logits
            loss_sum, _ = masked_loss_sum(logits, batch["input_ids"], batch["loss_mask"])
            return loss_sum / global_token_count, loss_sum

        (_, local_loss_sum), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        gradients = jax.lax.psum(gradients, "devices")
        gradient_norm = optax.global_norm(gradients)
        metrics = {"loss": jax.lax.psum(local_loss_sum, "devices") / global_token_count,
                   "answer_tokens": global_token_count, "gradient_norm": gradient_norm,
                   "learning_rate": schedule(state.step)}
        return state.apply_gradients(grads=gradients), metrics

    parallel_step = jax.pmap(training_step, axis_name="devices", donate_argnums=(0,), devices=backend.devices)
    print(json.dumps({"event": "training_started", "train_examples": len(examples), "total_steps": total_steps,
                      "steps_per_epoch": steps_per_epoch, "global_batch_size": batch_size,
                      "warmup_steps": warmup_steps, "source_digest": source_digest,
                      "maximum_example_tokens": max(len(row[0]) for row in examples)}), flush=True)
    start_time = time.monotonic()
    with (output_directory / "metrics.jsonl").open("x") as metric_handle:
        for epoch in range(start_epoch, configuration["epochs"]):
            order = np.random.default_rng(configuration["seed"] + epoch).permutation(len(examples))
            epoch_loss_sum = 0.0
            epoch_token_count = 0
            epoch_examples = 0
            for batch_number, offset in enumerate(range(0, len(examples), batch_size)):
                selected = order[offset : offset + batch_size]
                batch = make_batch(examples, selected, batch_size, configuration["maximum_sequence_length"],
                                   backend.tokenizer.pad_token_id)
                sharded_batch = {key: value.reshape(len(backend.devices), configuration["batch_size_per_device"], -1)
                                 for key, value in batch.items()}
                step_start = time.monotonic()
                state, device_metrics = parallel_step(state, sharded_batch)
                metrics = {key: float(np.asarray(value[0])) for key, value in device_metrics.items()}
                step_seconds = time.monotonic() - step_start
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"Non-finite training metrics: {metrics}")
                step = epoch * steps_per_epoch + batch_number + 1
                metrics.update({"event": "train_step", "step": step, "epoch": epoch + 1,
                                "batch_examples": len(selected), "step_seconds": step_seconds,
                                "examples_per_second": len(selected) / step_seconds,
                                "elapsed_seconds": time.monotonic() - start_time})
                metric_handle.write(json.dumps(metrics) + "\n")
                metric_handle.flush()
                if step <= 5 or step % 25 == 0:
                    print(json.dumps(metrics), flush=True)
                epoch_loss_sum += metrics["loss"] * metrics["answer_tokens"]
                epoch_token_count += int(metrics["answer_tokens"])
                epoch_examples += len(selected)
            if epoch_examples != len(examples):
                raise ValueError("Epoch example coverage is incomplete.")
            print(json.dumps({"event": "epoch_complete", "epoch": epoch + 1, "examples": epoch_examples,
                              "answer_token_loss": epoch_loss_sum / epoch_token_count}), flush=True)
            host_state = save_checkpoint(state, output_directory, epoch + 1, configuration, source_digest)
            del host_state

    host_parameters = jax.tree.map(lambda value: np.asarray(value[0]), state.params)
    export_directory = output_directory / "final-model"
    export_directory.mkdir()
    parameters = tied_parameters(host_parameters)
    backend.model.save_pretrained(export_directory, params=parameters)
    backend.tokenizer.save_pretrained(export_directory)
    pytorch_model = AutoModelForCausalLM.from_config(backend.config, attn_implementation="eager")
    pytorch_model = load_flax_weights_in_pytorch_model(pytorch_model, parameters)
    pytorch_model.tie_weights()
    pytorch_model.save_pretrained(export_directory, safe_serialization=True)
    manifest = {"status": "complete", "step": total_steps, "epochs": configuration["epochs"],
                "model": configuration["model"], "source_digest": source_digest,
                "files": [file_record(path) for path in sorted(export_directory.iterdir()) if path.is_file()]}
    (export_directory / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    mirror_directory(export_directory, configuration["artifact_bucket_prefix"])
    completed = {"status": "completed", "steps": total_steps, "epochs": configuration["epochs"],
                 "training_example_presentations": len(examples) * configuration["epochs"],
                 "model_directory": str(export_directory), "elapsed_seconds": time.monotonic() - start_time}
    (output_directory / "training-completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps(completed), flush=True)


if __name__ == "__main__":
    main()

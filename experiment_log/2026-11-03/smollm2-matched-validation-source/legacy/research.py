"""Run a bounded, validation-selected SmolLM2 GSM8K study on one remote TPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from datasets import load_dataset
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax_tpu_lm_eval import FlaxTPULanguageModel, trim_at_stop
from train_gsm8k import encode_example, file_record, make_batch, masked_loss_sum, mirror_directory, tied_parameters


def write_json(path: Path, value: Any) -> None:
    """Publish a new immutable JSON artifact."""
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def split_indices(count: int, validation_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Partition original training indices deterministically without replacement."""
    if not 0 < validation_count < count:
        raise ValueError("Validation must be a nonempty proper subset of training data.")
    order = np.random.default_rng(seed).permutation(count)
    return np.sort(order[validation_count:]), np.sort(order[:validation_count])


def normalize_answer(text: str) -> str:
    """Normalize a numeric GSM8K answer using the harness's ignored characters."""
    return text.replace(",", "").replace("$", "").removesuffix(".").strip().lower()


def score_response(response: str, target: str) -> tuple[bool, bool]:
    """Apply the pinned harness's strict and flexible numeric extraction rules."""
    target_number = normalize_answer(target.rsplit("#### ", 1)[-1])
    strict = re.findall(r"#### (\-?[0-9\.\,]+)", response)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    strict_number = normalize_answer(strict[0]) if strict else "[invalid]"
    flexible_number = normalize_answer("".join(flexible[-1])) if flexible else "[invalid]"
    return strict_number == target_number, flexible_number == target_number


def selection_key(metrics: dict[str, float]) -> tuple[float, float]:
    """Choose strict validation accuracy first and unweighted validation NLL second."""
    return metrics["validation/strict_accuracy"], -metrics["validation/loss"]


def weighted_example(row: dict[str, str], backend: FlaxTPULanguageModel,
                     length: int, final_answer_weight: float) -> tuple[list[int], list[float]]:
    """Optionally emphasize final numeric answers without weighting prompt or padding."""
    tokens, mask = encode_example(row["question"], row["answer"], backend.tokenizer, length)
    weights = list(map(float, mask))
    if final_answer_weight != 1.0:
        marker = row["answer"].rindex("####")
        prefix = "Question: " + row["question"] + "\nAnswer: " + row["answer"][:marker]
        prefix_tokens = backend.tokenizer.encode(prefix, add_special_tokens=False)
        if tokens[:len(prefix_tokens)] != prefix_tokens:
            raise ValueError("Final-answer weighting boundary changed tokenization.")
        weights[len(prefix_tokens):] = [final_answer_weight] * (len(tokens) - len(prefix_tokens))
    return tokens, weights


class StudyEngine:
    """Share verified model loading and fixed-shape evaluators across sequential trials."""

    def __init__(self, configuration: dict[str, Any], output_directory: Path) -> None:
        self.configuration = configuration
        self.output_directory = output_directory
        self.backend = FlaxTPULanguageModel(configuration["model"], configuration["model_revision"], batch_size=8)
        if len(self.backend.devices) != 8:
            raise ValueError("The approved study requires exactly eight local TPU chips.")
        del self.backend.replicated_parameters
        self.original_parameters = {"model": self.backend.parameters["model"]}
        dataset = load_dataset(configuration["dataset"], "main", revision=configuration["dataset_revision"], split="train")
        self.rows = [dict(row) for row in dataset]
        if len(self.rows) != 7473:
            raise ValueError("Unexpected GSM8K training set size.")
        train_indices, validation_indices = split_indices(len(self.rows), configuration["validation_examples"],
                                                          configuration["split_seed"])
        self.training_rows = [self.rows[int(index)] for index in train_indices]
        self.validation_rows = [self.rows[int(index)] for index in validation_indices]
        self.examples = [encode_example(row["question"], row["answer"], self.backend.tokenizer,
                                        configuration["sequence_length"]) for row in self.rows]
        self.training_examples = [self.examples[int(index)] for index in train_indices]
        self.validation_examples = [self.examples[int(index)] for index in validation_indices]
        split = {"train_indices": train_indices.tolist(), "validation_indices": validation_indices.tolist(),
                 "split_seed": configuration["split_seed"], "dataset_revision": configuration["dataset_revision"]}
        split_path = output_directory / "split.json"
        if split_path.exists():
            if json.loads(split_path.read_text()) != split:
                raise ValueError("Recovered study split differs from its immutable original.")
        else:
            write_json(split_path, split)
        self.parallel_loss = jax.pmap(self.loss_on_device, axis_name="devices", devices=self.backend.devices)
        self.parallel_generate = jax.pmap(self.generate_on_device, devices=self.backend.devices)

    def loss_on_device(self, parameters: dict[str, Any], batch: dict[str, jax.Array]) -> jax.Array:
        """Return globally summed unweighted answer NLL and token count in evaluation mode."""
        logits = self.backend.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                                    params=tied_parameters(parameters), train=False).logits
        loss_sum, token_count = masked_loss_sum(logits, batch["input_ids"], batch["loss_mask"])
        return jax.lax.psum(jnp.stack([loss_sum, token_count]), "devices")

    def generate_on_device(self, parameters: dict[str, Any], input_ids: jax.Array,
                           attention_mask: jax.Array) -> jax.Array:
        """Generate the same zero-shot validation protocol at every checkpoint."""
        return self.backend.model.generate(input_ids, attention_mask=attention_mask,
            params=tied_parameters(parameters), max_new_tokens=self.configuration["generation_tokens"],
            do_sample=False, num_beams=1, pad_token_id=self.backend.tokenizer.pad_token_id,
            eos_token_id=self.backend.tokenizer.eos_token_id).sequences

    def loss(self, parameters: dict[str, Any], examples: list) -> dict[str, float]:
        """Measure exact full-split answer-token NLL with no truncation or tail duplication."""
        total_loss = 0.0
        total_tokens = 0
        sequence_length = max(self.configuration["sequence_length"],
                              math.ceil(max(len(example[0]) for example in examples) / 128) * 128)
        for offset in range(0, len(examples), 32):
            indices = np.arange(offset, min(offset + 32, len(examples)))
            batch = make_batch(examples, indices, 32, sequence_length,
                               self.backend.tokenizer.pad_token_id)
            batch = {key: value.reshape(8, 4, -1) for key, value in batch.items()}
            values = np.asarray(self.parallel_loss(parameters, batch)[0])
            total_loss += float(values[0])
            total_tokens += int(values[1])
        return {"loss": total_loss / total_tokens, "answer_tokens": total_tokens, "examples": len(examples)}

    def generate_scores(self, parameters: dict[str, Any], rows: list[dict[str, str]],
                        path: Path) -> dict[str, float]:
        """Save local per-question generations and score every held-out validation example."""
        tokenizer = self.backend.tokenizer
        prompt_width = self.configuration["generation_prompt_width"]
        prompts = [tokenizer.encode("Question: " + row["question"] + "\nAnswer:", add_special_tokens=False)
                   for row in rows]
        if max(map(len, prompts)) > prompt_width:
            raise ValueError("A generation prompt exceeds the fixed width; truncation is disabled.")
        correct = np.zeros(2, dtype=np.int64)
        with path.open("x") as handle:
            for offset in range(0, len(rows), 64):
                selected = prompts[offset:offset + 64]
                input_ids = np.full((64, prompt_width), tokenizer.pad_token_id, dtype=np.int32)
                attention = np.zeros_like(input_ids)
                for index in range(64):
                    tokens = selected[min(index, len(selected) - 1)]
                    input_ids[index, -len(tokens):] = tokens
                    attention[index, -len(tokens):] = 1
                generated = np.asarray(self.parallel_generate(parameters, input_ids.reshape(8, 8, -1),
                                                              attention.reshape(8, 8, -1))).reshape(64, -1)
                texts = tokenizer.batch_decode(generated[:len(selected), prompt_width:], skip_special_tokens=False)
                for index, text in enumerate(texts):
                    response = trim_at_stop(text, ["Question:", "</s>", "<|im_end|>", tokenizer.eos_token])
                    for special_token in tokenizer.all_special_tokens:
                        response = response.replace(special_token, "")
                    scores = score_response(response, rows[offset + index]["answer"])
                    correct += np.asarray(scores, dtype=np.int64)
                    handle.write(json.dumps({"index": offset + index, "response": response,
                                             "strict_correct": scores[0], "flexible_correct": scores[1]}) + "\n")
        return {"strict_accuracy": int(correct[0]) / len(rows), "flexible_accuracy": int(correct[1]) / len(rows),
                "strict_correct": int(correct[0]), "flexible_correct": int(correct[1]), "examples": len(rows)}

    def save_state(self, state: TrainState, directory: Path, metadata: dict[str, Any]) -> None:
        """Durably mirror a completed best checkpoint with optimizer state and checksums."""
        directory.mkdir()
        state_path = directory / "training_state.msgpack"
        with state_path.open("xb") as handle:
            handle.write(serialization.to_bytes(state))
            handle.flush()
            os.fsync(handle.fileno())
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "complete.json", {"status": "complete", "step": int(state.step),
                   "files": [file_record(state_path), file_record(directory / "metadata.json")]})
        mirror_directory(directory, self.configuration["artifact_bucket_prefix"])

    def verify_loss(self) -> None:
        """Compare distributed masked NLL with an independent PyTorch reference."""
        import torch
        from transformers import AutoModelForCausalLM

        parameters = jax.device_put_replicated(self.original_parameters, self.backend.devices)
        selected = self.validation_examples[:3]
        measured = self.loss(parameters, selected)
        reference = AutoModelForCausalLM.from_pretrained(self.configuration["model"],
            revision=self.configuration["model_revision"], torch_dtype=torch.float32, attn_implementation="eager")
        reference.eval()
        batch = make_batch(selected, np.arange(3), 3, self.configuration["sequence_length"],
                           self.backend.tokenizer.pad_token_id)
        with torch.inference_mode():
            tokens = torch.from_numpy(batch["input_ids"]).long()
            logits = reference(tokens, attention_mask=torch.from_numpy(batch["attention_mask"]), use_cache=False).logits
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                       tokens[:, 1:].reshape(-1), reduction="none")
            mask = torch.from_numpy(batch["loss_mask"][:, 1:].reshape(-1))
            reference_loss = float((losses * mask).sum() / mask.sum())
        difference = abs(measured["loss"] - reference_loss)
        if difference > 1e-4 or measured["answer_tokens"] != int(mask.sum()):
            raise ValueError(f"TPU/reference loss mismatch: {measured}, {reference_loss}")
        write_json(self.output_directory / "loss-parity.json", {"status": "passed", "tpu": measured,
                   "pytorch_loss": reference_loss, "absolute_difference": difference, "tolerance": 1e-4})
        del reference, parameters
        gc.collect()

    def trial(self, configuration: dict[str, Any], deadline: float,
              test_rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Train one candidate from the base model and select only by validation evidence."""
        directory = self.output_directory / configuration["trial_id"]
        directory.mkdir()
        write_json(directory / "config.json", configuration)
        final_fit = configuration.get("final_fit", False)
        rows = self.rows if final_fit else self.training_rows
        evaluation_examples = self.examples if final_fit else self.training_examples
        examples = [weighted_example(row, self.backend, self.configuration["sequence_length"],
                                    configuration["final_answer_weight"]) for row in rows]
        test_examples = None if test_rows is None else [encode_example(row["question"], row["answer"],
                    self.backend.tokenizer, self.backend.config.max_position_embeddings) for row in test_rows]
        batch_size = configuration["batch_size"]
        steps_per_epoch = math.ceil(len(examples) / batch_size)
        total_steps = steps_per_epoch * configuration["schedule_epochs"]
        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=configuration["learning_rate"],
            warmup_steps=max(1, math.ceil(total_steps * configuration["warmup_fraction"])), decay_steps=total_steps,
            end_value=configuration["learning_rate"] * configuration["final_learning_rate_ratio"])
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, b1=0.9, b2=0.999,
                                eps=1e-8, weight_decay=configuration["weight_decay"]))
        with jax.default_device(jax.devices("cpu")[0]):
            host_state = TrainState.create(apply_fn=self.backend.model.__call__, params=self.original_parameters, tx=optimizer)
        state = jax.device_put_replicated(host_state, self.backend.devices)
        del host_state

        def training_step(current: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
            """Apply globally token-weighted gradients to the single tied parameter tree."""
            count = jax.lax.psum(jnp.sum(batch["loss_mask"][:, 1:]), "devices")

            def objective(parameters: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
                """Accumulate both tied-embedding gradient paths inside differentiation."""
                random_key = jax.random.fold_in(jax.random.PRNGKey(configuration["seed"]), current.step)
                random_key = jax.random.fold_in(random_key, jax.lax.axis_index("devices"))
                logits = self.backend.model(batch["input_ids"], attention_mask=batch["attention_mask"],
                    params=tied_parameters(parameters), train=True, dropout_rng=random_key).logits
                numerator, _ = masked_loss_sum(logits, batch["input_ids"], batch["loss_mask"])
                return numerator / count, numerator

            (_, local_sum), gradients = jax.value_and_grad(objective, has_aux=True)(current.params)
            gradients = jax.lax.psum(gradients, "devices")
            metrics = {"update_loss": jax.lax.psum(local_sum, "devices") / count,
                       "gradient_norm": optax.global_norm(gradients), "learning_rate": schedule(current.step),
                       "weighted_answer_tokens": count}
            return current.apply_gradients(grads=gradients), metrics

        parallel_step = jax.pmap(training_step, axis_name="devices", donate_argnums=(0,), devices=self.backend.devices)
        best_metrics: dict[str, float] | None = None
        best_state = None
        best_loss = float("inf")
        best_loss_epoch = 0
        best_epoch = 0
        start_time = time.monotonic()
        with (directory / "events.jsonl").open("x") as handle:
            def emit(metrics: dict[str, float]) -> None:
                """Append only scalar scientific metrics to the W&B bridge input."""
                handle.write(json.dumps({"event": "metrics", "metrics": metrics}) + "\n")
                handle.flush()
                print(json.dumps({"trial_id": configuration["trial_id"], **metrics}), flush=True)

            initial = self.loss(state.params, evaluation_examples)
            initial_metrics = {"epoch": 0, "step": 0, "train/loss": initial["loss"]}
            if not final_fit:
                initial_metrics["validation/loss"] = self.loss(state.params, self.validation_examples)["loss"]
            if test_examples is not None:
                initial_metrics["test/loss"] = self.loss(state.params, test_examples)["loss"]
            emit(initial_metrics)
            for epoch in range(configuration["epochs"]):
                order = np.random.default_rng(configuration["seed"] + epoch).permutation(len(examples))
                for batch_number, offset in enumerate(range(0, len(examples), batch_size)):
                    selected = order[offset:offset + batch_size]
                    batch = make_batch(examples, selected, batch_size, self.configuration["sequence_length"],
                                       self.backend.tokenizer.pad_token_id)
                    batch = {key: value.reshape(8, batch_size // 8, -1) for key, value in batch.items()}
                    state, device_metrics = parallel_step(state, batch)
                    metrics = {"train/" + key: float(np.asarray(value[0])) for key, value in device_metrics.items()}
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError("Nonfinite training metric.")
                    step = epoch * steps_per_epoch + batch_number + 1
                    metrics.update({"step": step, "epoch": epoch + (batch_number + 1) / steps_per_epoch})
                    handle.write(json.dumps({"event": "metrics", "metrics": metrics}) + "\n")
                    handle.flush()
                    if step <= 3 or step % 100 == 0:
                        print(json.dumps({"trial_id": configuration["trial_id"], **metrics}), flush=True)
                metrics = {"epoch": epoch + 1, "step": step, "elapsed_seconds": time.monotonic() - start_time}
                metrics.update({"train/" + key: value for key, value in self.loss(state.params, evaluation_examples).items()})
                if not final_fit:
                    metrics.update({"validation/" + key: value for key, value in self.loss(state.params, self.validation_examples).items()})
                    scores = self.generate_scores(state.params, self.validation_rows, directory / f"validation-epoch{epoch + 1:02d}.jsonl")
                    metrics.update({"validation/" + key: value for key, value in scores.items()})
                if test_examples is not None:
                    metrics.update({"test/" + key: value for key, value in self.loss(state.params, test_examples).items()})
                emit(metrics)
                tracked_loss = metrics["train/loss"] if final_fit else metrics["validation/loss"]
                if tracked_loss < best_loss:
                    best_loss, best_loss_epoch = tracked_loss, epoch + 1
                if final_fit or best_metrics is None or selection_key(metrics) > selection_key(best_metrics):
                    best_metrics = dict(metrics)
                    best_epoch = epoch + 1
                    best_state = jax.tree.map(lambda value: np.asarray(value[0]), state)
                if not final_fit and epoch + 1 >= configuration["minimum_epochs"]:
                    stagnant = epoch + 1 - max(best_epoch, best_loss_epoch) >= configuration["patience"]
                    if stagnant or time.monotonic() >= deadline:
                        break
            assert best_state is not None and best_metrics is not None
            summary = {"status": "completed", "trial_id": configuration["trial_id"], "configuration": configuration,
                       "best_epoch": best_epoch, "best_metrics": best_metrics, "best_loss": best_loss,
                       "best_loss_epoch": best_loss_epoch, "completed_epochs": epoch + 1,
                       "elapsed_seconds": time.monotonic() - start_time,
                       "study_configuration": self.configuration,
                       "source_digest": hashlib.sha256(Path("source.sha256").read_bytes()).hexdigest()}
            checkpoint = self.output_directory / (configuration["trial_id"] + "-checkpoint")
            self.save_state(best_state, checkpoint, summary)
            summary["checkpoint"] = str(checkpoint)
            write_json(directory / "summary.json", summary)
            handle.write(json.dumps({"event": "completed", "summary": {"best_epoch": best_epoch,
                "best_validation_loss": best_loss, "completed_epochs": epoch + 1,
                "selected_validation_loss": best_metrics.get("validation/loss"),
                "selected_validation_accuracy": best_metrics.get("validation/strict_accuracy", 0.0)}}) + "\n")
            handle.flush()
        del state, best_state, parallel_step
        gc.collect()
        return summary


def candidate_configuration(index: int, learning_rate: float, batch_size: int,
                            **overrides: Any) -> dict[str, Any]:
    """Create a complete deterministic candidate configuration before its launch."""
    return {"trial_id": f"trial-{index:02d}", "learning_rate": learning_rate, "batch_size": batch_size,
            "epochs": 12, "schedule_epochs": 12, "minimum_epochs": 3, "patience": 3, "seed": 1234,
            "warmup_fraction": 0.03, "final_learning_rate_ratio": 0.1, "weight_decay": 0.01,
            "final_answer_weight": 1.0, "final_fit": False, **overrides}


def main() -> None:
    """Explore candidates under one compute budget, lock the choice, then measure test loss."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text())
    output_directory = Path(configuration["output_directory"])
    if not str(output_directory).startswith("/home/kristinelu/smollm2-gsm8k/"):
        raise ValueError("Study jobs must run on the authorized remote node-local storage.")
    output_directory.mkdir(exist_ok=True)
    start_marker = output_directory / "study-started.json"
    if not start_marker.exists():
        write_json(start_marker, {"unix_seconds": time.time(), "configuration": configuration})
    start_wall = json.loads(start_marker.read_text())["unix_seconds"]
    remaining_seconds = start_wall + configuration["budget_seconds"] - time.time()
    final_deadline = time.monotonic() + remaining_seconds
    search_deadline = final_deadline - configuration["finalization_reserve_seconds"]
    engine = StudyEngine(configuration, output_directory)
    if not (output_directory / "loss-parity.json").exists():
        engine.verify_loss()
    summaries = []
    # SmolTulu: Higher Learning Rate to Batch Size Ratios Can Lead to Better Reasoning in SLMs https://arxiv.org/abs/2412.08347
    initial = [(1e-5, 32), (3e-5, 32), (5e-5, 32), (1e-4, 32), (3e-4, 32), (1e-3, 32)]
    index = 0
    while index < configuration["maximum_trials"]:
        index += 1
        if time.monotonic() > search_deadline - 480 or shutil.disk_usage(output_directory).free < 12 * 1024**3:
            break
        if index <= len(initial):
            learning_rate, batch_size = initial[index - 1]
            candidate = candidate_configuration(index, learning_rate, batch_size)
        else:
            best = max(summaries, key=lambda item: selection_key(item["best_metrics"]))
            base = best["configuration"]
            variants = [dict(batch_size=8), dict(batch_size=16), dict(batch_size=64),
                        dict(learning_rate=base["learning_rate"] * 0.5),
                        dict(learning_rate=min(base["learning_rate"] * 2, 0.002)),
                        dict(weight_decay=0.0), dict(weight_decay=0.1),
                        dict(final_answer_weight=2.0), dict(final_answer_weight=4.0),
                        dict(warmup_fraction=0.1), dict(final_learning_rate_ratio=0.01),
                        dict(epochs=20, schedule_epochs=20), dict(seed=2345), dict(seed=3456)]
            variant_index = index - len(initial) - 1
            if variant_index >= len(variants):
                break
            candidate = {**base, **variants[variant_index], "trial_id": f"trial-{index:02d}", "final_fit": False}
        summary_path = output_directory / candidate["trial_id"] / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))
            continue
        summary = engine.trial(candidate, search_deadline)
        summaries.append(summary)
        write_json(output_directory / f"leaderboard-{index:02d}.json", sorted(summaries,
                   key=lambda item: selection_key(item["best_metrics"]), reverse=True))
    if not summaries:
        raise RuntimeError("No completed candidate is available for selection.")
    winner = max(summaries, key=lambda item: selection_key(item["best_metrics"]))
    selection = {"criterion": "maximum validation strict accuracy; minimum validation NLL breaks ties",
                 "test_used_for_selection": False, "winner": winner, "completed_trials": len(summaries),
                 "selection_unix_seconds": time.time()}
    write_json(output_directory / "selection-locked.json", selection)
    test_dataset = load_dataset(configuration["dataset"], "main", revision=configuration["dataset_revision"], split="test")
    test_rows = [dict(row) for row in test_dataset]
    if len(test_rows) != 1319:
        raise ValueError("Unexpected test set size.")
    final_configuration = {**winner["configuration"], "trial_id": "final-fit", "final_fit": True,
                           "epochs": winner["best_epoch"], "seed": 1234}
    final = engine.trial(final_configuration, final_deadline, test_rows=test_rows)
    write_json(output_directory / "study-completed.json", {"status": "completed", "selection": selection,
               "final": final, "elapsed_seconds": time.time() - start_wall})
    print("STUDY_COMPLETED", flush=True)


if __name__ == "__main__":
    main()

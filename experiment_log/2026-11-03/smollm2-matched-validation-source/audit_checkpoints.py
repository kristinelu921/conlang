"""Measure both validation prompt formats for saved training-only TPU checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, TYPE_CHECKING

from matched_protocol import (ComputeBudget, PromptExample, Question, demonstration_pool,
                              prepare_prompt, protocol_digest, validate_partition)

if TYPE_CHECKING:
    from research import StudyEngine


def read_object(path: Path) -> dict[str, Any]:
    """Read the single supported JSON object shape at the filesystem boundary."""
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def save_object(path: Path, value: dict[str, Any]) -> None:
    """Preserve an immutable JSON record without replacing any earlier evidence."""
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_execution_receipt(receipt: dict[str, Any], protocol: dict[str, Any],
                             source_directory: Path, hostname: str,
                             now_unix_seconds: float) -> ComputeBudget:
    """Require recorded budget approval and the exact verified remote host before ML imports."""
    hardware = protocol["hardware"]
    if receipt["approval_status"] != "approved" or receipt["study_id"] != protocol["study_id"]:
        raise PermissionError("The new study has no matching explicit budget approval.")
    for key in ("project", "zone", "name", "type"):
        if receipt["hardware"][key] != hardware[key]:
            raise PermissionError("Execution receipt does not name the authorized TPU.")
    if "kristine" not in hardware["name"].lower() or hardware["type"] != "v5litepod-8":
        raise PermissionError("This continuation permits only the single specified kristine Spot v5e-8.")
    if receipt["maximum_concurrent_tpus"] != 1 or receipt["verified_hostname"] != hostname:
        raise PermissionError("This process is not on the preflight-verified single remote host.")
    expected_source = Path("/home/kristinelu/smollm2-gsm8k/matched-validation-20261103/source")
    if source_directory.resolve() != expected_source:
        raise PermissionError("Experiment code must run from the dedicated VM's node-local source snapshot.")
    if not receipt["spot"] or receipt["development_machine"]:
        raise PermissionError("A development machine or unapproved allocation cannot run this study.")
    budget = ComputeBudget(receipt["study_started_unix_seconds"], receipt["authorized_seconds"])
    if budget.remaining(now_unix_seconds) <= protocol["budget"]["finalization_reserve_seconds"]:
        raise TimeoutError("No authorized validation-search time remains before the finalization reserve.")
    return budget


def verify_source_manifest(directory: Path) -> str:
    """Check every staged source file before importing the inherited TPU implementation."""
    manifest = directory / "source.sha256"
    for line in manifest.read_text().splitlines():
        expected_hash, name = line.split("  ", 1)
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("Source manifest contains a missing or nonlocal file.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Staged source checksum mismatch: {name}")
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def require_search_time(budget: ComputeBudget, reserve_seconds: float) -> None:
    """Refuse further model work when the fixed finalization reserve has been reached."""
    if budget.remaining(time.time()) <= reserve_seconds:
        raise TimeoutError("Validation audit reached its reserved finalization boundary.")


def evaluate_loss(engine: StudyEngine, parameters: Any, examples: list[PromptExample],
                  budget: ComputeBudget, reserve_seconds: float) -> dict[str, float | int]:
    """Accumulate query-answer NLL with exact token weighting and per-batch deadline checks."""
    import numpy as np
    from train_gsm8k import make_batch

    maximum_length = max(len(example.input_tokens) for example in examples)
    sequence_length = max(640, math.ceil(maximum_length / 128) * 128)
    if sequence_length > engine.backend.config.max_position_embeddings:
        raise ValueError("Loss batch exceeds context; truncation is disabled.")
    encoded = [(list(example.input_tokens), list(example.loss_mask)) for example in examples]
    total_loss = 0.0
    total_tokens = 0
    for offset in range(0, len(encoded), 32):
        require_search_time(budget, reserve_seconds)
        indices = np.arange(offset, min(offset + 32, len(encoded)))
        batch = make_batch(encoded, indices, 32, sequence_length, engine.backend.tokenizer.pad_token_id)
        batch = {key: value.reshape(8, 4, -1) for key, value in batch.items()}
        values = np.asarray(engine.parallel_loss(parameters, batch)[0])
        total_loss += float(values[0])
        total_tokens += int(values[1])
    if not math.isfinite(total_loss) or total_tokens <= 0:
        raise ValueError("Validation loss is nonfinite or has no supervised tokens.")
    return {"nll": total_loss / total_tokens, "answer_tokens": total_tokens, "examples": len(examples)}


def evaluate_generations(engine: StudyEngine, parameters: Any, examples: list[PromptExample],
                         targets: list[str], output_path: Path, budget: ComputeBudget,
                         reserve_seconds: float) -> dict[str, float | int]:
    """Score all validation queries without including duplicate final-batch padding rows."""
    import numpy as np
    from flax_tpu_lm_eval import trim_at_stop
    from research import score_response

    if len(examples) != len(targets) or not examples:
        raise ValueError("Validation targets and nonempty prompts must have equal length.")
    tokenizer = engine.backend.tokenizer
    prompt_width = math.ceil(max(len(example.prompt_tokens) for example in examples) / 128) * 128
    if all(not example.demonstration_indices for example in examples):
        prompt_width = max(512, prompt_width)
    if prompt_width + 256 > engine.backend.config.max_position_embeddings:
        raise ValueError("Generation exceeds context; truncation is disabled.")
    correct = np.zeros(2, dtype=np.int64)
    with output_path.open("x") as handle:
        for offset in range(0, len(examples), 64):
            require_search_time(budget, reserve_seconds)
            selected = examples[offset:offset + 64]
            input_ids = np.full((64, prompt_width), tokenizer.pad_token_id, dtype=np.int32)
            attention = np.zeros_like(input_ids)
            for index in range(64):
                tokens = selected[min(index, len(selected) - 1)].prompt_tokens
                input_ids[index, -len(tokens):] = tokens
                attention[index, -len(tokens):] = 1
            generated = np.asarray(engine.parallel_generate(parameters, input_ids.reshape(8, 8, -1),
                                                            attention.reshape(8, 8, -1))).reshape(64, -1)
            texts = tokenizer.batch_decode(generated[:len(selected), prompt_width:], skip_special_tokens=False)
            for index, text in enumerate(texts):
                response = trim_at_stop(text, ["Question:", "</s>", "<|im_end|>", tokenizer.eos_token])
                for special_token in tokenizer.all_special_tokens:
                    response = response.replace(special_token, "")
                scores = score_response(response, targets[offset + index])
                correct += np.asarray(scores, dtype=np.int64)
                handle.write(json.dumps({"source_index": selected[index].query_index,
                    "demonstration_indices": selected[index].demonstration_indices,
                    "response": response, "strict_correct": scores[0], "flexible_correct": scores[1]}) + "\n")
            handle.flush()
    return {"strict_correct": int(correct[0]), "flexible_correct": int(correct[1]),
            "strict_accuracy": int(correct[0]) / len(examples),
            "flexible_accuracy": int(correct[1]) / len(examples), "examples": len(examples)}


def load_verified_parameters(checkpoint: Path, summary: dict[str, Any],
                             protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a completed training-only checkpoint and restore its concrete Flax parameter tree."""
    from flax import serialization
    from train_gsm8k import file_record

    manifest = read_object(checkpoint / "complete.json")
    if manifest["status"] != "complete":
        raise ValueError("Checkpoint has no completion marker.")
    expected_names = {"training_state.msgpack", "metadata.json"}
    if {record["name"] for record in manifest["files"]} != expected_names or len(manifest["files"]) != 2:
        raise ValueError("Checkpoint manifest has unexpected files.")
    for record in manifest["files"]:
        if file_record(checkpoint / record["name"]) != record:
            raise ValueError("Checkpoint size or SHA-256 verification failed.")
    metadata = read_object(checkpoint / "metadata.json")
    if metadata["configuration"]["final_fit"] or summary["configuration"]["final_fit"]:
        raise ValueError("Full-data refits are ineligible for validation-based selection.")
    if metadata["trial_id"] != summary["trial_id"] or metadata["best_epoch"] != summary["best_epoch"]:
        raise ValueError("Checkpoint identity or selected epoch differs from its trial summary.")
    for key in ("model", "model_revision", "dataset", "dataset_revision"):
        if metadata["study_configuration"][key] != protocol[key]:
            raise ValueError("Checkpoint uses a different model or dataset revision.")
    if (metadata["study_configuration"]["split_seed"] != protocol["partition"]["split_seed"]
            or metadata["study_configuration"]["validation_examples"] != protocol["partition"]["validation_examples"]):
        raise ValueError("Checkpoint uses a different validation partition.")
    state = serialization.msgpack_restore((checkpoint / "training_state.msgpack").read_bytes())
    assert isinstance(state, dict)
    if int(state["step"]) != manifest["step"] or int(state["step"]) != metadata["best_metrics"]["step"]:
        raise ValueError("Checkpoint step does not match its completed selected epoch.")
    return state["params"], manifest


def verify_long_context(engine: StudyEngine, prompts: dict[int, list[PromptExample]],
                         targets: list[str], output: Path, budget: ComputeBudget,
                         reserve_seconds: float) -> dict[str, Any]:
    """Compare both prompt formats with independent PyTorch loss and long-position logits."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from train_gsm8k import make_batch, tied_parameters

    require_search_time(budget, reserve_seconds)
    reference = AutoModelForCausalLM.from_pretrained(engine.configuration["model"],
        revision=engine.configuration["model_revision"], torch_dtype=torch.float32,
        attn_implementation="eager")
    reference.eval()
    parameters = jax.device_put_replicated(engine.original_parameters, engine.backend.devices)

    def selected_logits(device_parameters: Any, batch: dict[str, Any], positions: Any) -> Any:
        """Return only three probe positions per row to keep host transfers bounded."""
        logits = engine.backend.model(batch["input_ids"], attention_mask=batch["attention_mask"],
            params=tied_parameters(device_parameters), train=False).logits
        return logits[jnp.arange(logits.shape[0])[:, None], positions]

    parallel_logits = jax.pmap(selected_logits, devices=engine.backend.devices)
    report: dict[str, Any] = {"status": "passed", "loss_tolerance": 1e-4,
        "logit_tolerance": 0.02, "model_revision": engine.configuration["model_revision"],
        "test_used_for_selection": False, "protocols": {}}
    for shots, label in ((0, "zero_shot"), (5, "five_shot")):
        require_search_time(budget, reserve_seconds)
        indices = sorted(range(len(prompts[shots])), key=lambda index: len(prompts[shots][index].input_tokens))
        selected_indices = [indices[0], indices[len(indices) // 2], indices[-1]]
        selected = [prompts[shots][index] for index in selected_indices]
        encoded = [(list(example.input_tokens), list(example.loss_mask)) for example in selected]
        sequence_length = max(640, math.ceil(max(len(example.input_tokens) for example in selected) / 128) * 128)
        batch = make_batch(encoded, np.arange(3), 8, sequence_length, engine.backend.tokenizer.pad_token_id)
        positions = np.zeros((8, 3), dtype=np.int32)
        for index, example in enumerate(selected):
            positions[index] = [0, len(example.prompt_tokens) - 1, len(example.input_tokens) - 2]
        sharded = {key: value.reshape(8, 1, -1) for key, value in batch.items()}
        observed_logits = np.asarray(parallel_logits(parameters, sharded, positions.reshape(8, 1, 3))).reshape(8, 3, -1)[:3]
        with torch.inference_mode():
            tokens = torch.from_numpy(batch["input_ids"][:3]).long()
            reference_logits = reference(tokens, attention_mask=torch.from_numpy(batch["attention_mask"][:3]),
                                         use_cache=False).logits
            expected_logits = reference_logits[torch.arange(3)[:, None], torch.from_numpy(positions[:3]).long()].numpy()
            losses = torch.nn.functional.cross_entropy(reference_logits[:, :-1].reshape(-1, reference_logits.shape[-1]),
                                                       tokens[:, 1:].reshape(-1), reduction="none")
            mask = torch.from_numpy(batch["loss_mask"][:3, 1:].reshape(-1))
            expected_loss = float((losses * mask).sum() / mask.sum())
        observed_loss = evaluate_loss(engine, parameters, selected, budget, reserve_seconds)
        loss_difference = abs(observed_loss["nll"] - expected_loss)
        logit_difference = float(np.max(np.abs(observed_logits - expected_logits)))
        if (not math.isfinite(logit_difference) or logit_difference > 0.02 or loss_difference > 1e-4
                or observed_loss["answer_tokens"] != int(mask.sum())):
            raise ValueError(f"{label} TPU/PyTorch parity failed: NLL delta {loss_difference}, logits {logit_difference}")
        generation_indices = sorted(range(len(prompts[shots])),
            key=lambda index: len(prompts[shots][index].prompt_tokens), reverse=True)[:64]
        generation_prompts = [prompts[shots][index] for index in generation_indices]
        generation_targets = [targets[index] for index in generation_indices]
        compilation_started = time.monotonic()
        evaluate_generations(engine, parameters, generation_prompts, generation_targets,
            output / (label + "-compile-generations.jsonl"), budget, reserve_seconds)
        compilation_and_generation_seconds = time.monotonic() - compilation_started
        warm_started = time.monotonic()
        evaluate_generations(engine, parameters, generation_prompts, generation_targets,
            output / (label + "-warm-generations.jsonl"), budget, reserve_seconds)
        warm_generation_seconds = time.monotonic() - warm_started
        report["protocols"][label] = {"query_source_indices": [example.query_index for example in selected],
            "prompt_lengths": [len(example.prompt_tokens) for example in selected],
            "sequence_length": sequence_length, "tpu_nll": observed_loss["nll"],
            "pytorch_nll": expected_loss, "absolute_nll_difference": loss_difference,
            "maximum_absolute_logit_difference": logit_difference,
            "generation_batch_size": 64, "compile_and_generation_seconds": compilation_and_generation_seconds,
            "warm_generation_seconds": warm_generation_seconds}
        del reference_logits, observed_logits, expected_logits
    del reference, parameters
    return report


def main() -> None:
    """Audit saved checkpoints only after recorded approval and exact remote-source verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    protocol = read_object(arguments.protocol)
    receipt = read_object(arguments.execution_receipt)
    source = Path(__file__).resolve().parent
    budget = verify_execution_receipt(receipt, protocol, source, socket.gethostname(), time.time())
    source_digest = verify_source_manifest(source)
    output = arguments.output_directory.resolve()
    expected_parent = source.parent / "output"
    expected_prefix = "preflight-" if arguments.preflight_only else "audit-"
    if output.parent != expected_parent or not output.name.startswith(expected_prefix):
        raise ValueError("Outputs require a new node-local study output/preflight-* or output/audit-* directory.")
    output.mkdir(parents=True, exist_ok=False)
    reserve_seconds = protocol["budget"]["finalization_reserve_seconds"]
    save_object(output / "started.json", {"status": "running", "source_manifest_sha256": source_digest,
        "receipt_sha256": hashlib.sha256(arguments.execution_receipt.read_bytes()).hexdigest(),
        "protocol_file_sha256": hashlib.sha256(arguments.protocol.read_bytes()).hexdigest(),
        "started_unix_seconds": time.time(), "test_used_for_selection": False})
    sys.path.insert(0, str(source / "legacy"))
    import jax
    import numpy as np
    from research import StudyEngine

    previous_output = Path("/home/kristinelu/smollm2-gsm8k/research-20261102/output")
    original_configuration = read_object(previous_output.parent / "study_config.json")
    for key in ("model", "model_revision", "dataset", "dataset_revision"):
        if original_configuration[key] != protocol[key]:
            raise ValueError("Original model/data configuration differs from the declared protocol.")
    if (original_configuration["split_seed"] != protocol["partition"]["split_seed"]
            or original_configuration["validation_examples"] != protocol["partition"]["validation_examples"]):
        raise ValueError("Original split configuration differs from the declared protocol.")
    configuration = {**original_configuration, "output_directory": str(output)}
    engine = StudyEngine(configuration, output)
    original_split = read_object(previous_output / "split.json")
    if read_object(output / "split.json") != original_split:
        raise ValueError("Reconstructed split differs from the prior immutable split.")
    training = [Question(index, **engine.rows[index]) for index in original_split["train_indices"]]
    validation = [Question(index, **engine.rows[index]) for index in original_split["validation_indices"]]
    validate_partition(training, validation, len(engine.rows))
    pool = demonstration_pool(training)
    prompts = {shots: [prepare_prompt(row, pool, engine.backend.tokenizer, shots,
        protocol["prompt_protocol"]["validation_demonstration_seed"], 0,
        engine.backend.config.max_position_embeddings,
        protocol["prompt_protocol"]["answer_token_reserve"],
        protocol["prompt_protocol"]["maximum_demonstration_sampling_attempts"])
        for row in validation] for shots in (0, 5)}
    targets = [row.answer for row in validation]
    digest = protocol_digest(prompts[0], prompts[5], targets)
    save_object(output / "prompt-manifest.json", {"protocol_sha256": digest,
        "validation_source_indices": [row.source_index for row in validation],
        "demonstration_pool_indices": [row.source_index for row in training],
        "five_shot_demonstrations": [asdict(example) for example in prompts[5]],
        "test_used_for_selection": False})
    if arguments.preflight_only:
        try:
            parity = verify_long_context(engine, prompts, targets, output, budget, reserve_seconds)
            save_object(output / "preflight-passed.json", {**parity, "protocol_sha256": digest,
                "finished_unix_seconds": time.time()})
        except BaseException as error:
            save_object(output / "preflight-failed.json", {"status": "failed",
                "error_type": type(error).__name__, "error": str(error), "finished_unix_seconds": time.time()})
            raise
        return
    summaries = [read_object(path) for path in sorted(previous_output.glob("*/summary.json"))]
    summaries = [summary for summary in summaries if not summary["configuration"]["final_fit"]]
    priority = ["trial-02", "trial-03", "trial-04", "loss-probe-01", "trial-13", "trial-10", "trial-08"]
    summaries.sort(key=lambda summary: (priority.index(summary["trial_id"]) if summary["trial_id"] in priority
                                       else len(priority), summary["trial_id"]))
    if len(summaries) != 19 or len({summary["trial_id"] for summary in summaries}) != 19:
        raise ValueError("Expected the completed study's 19 unique training-only candidate checkpoints.")
    results = []
    try:
        for summary in summaries:
            require_search_time(budget, reserve_seconds)
            identity = summary["trial_id"]
            directory = output / identity
            directory.mkdir()
            checkpoint = Path(summary["checkpoint"]).resolve()
            if checkpoint != previous_output / (identity + "-checkpoint"):
                raise ValueError("Unexpected source checkpoint location.")
            parameters, manifest = load_verified_parameters(checkpoint, summary, protocol)
            if any(np.asarray(leaf).dtype != np.float32 for leaf in jax.tree.leaves(parameters)):
                raise ValueError("Checkpoint parameter precision changed.")
            replicated = jax.device_put_replicated(parameters, engine.backend.devices)
            started = time.monotonic()
            result: dict[str, Any] = {"trial_id": identity, "epoch": summary["best_epoch"],
                "protocol_sha256": digest, "checkpoint": str(checkpoint), "checkpoint_manifest": manifest,
                "test_used_for_selection": False}
            for shots, label in ((0, "zero_shot"), (5, "five_shot")):
                result[label] = {**evaluate_loss(engine, replicated, prompts[shots], budget, reserve_seconds),
                    **evaluate_generations(engine, replicated, prompts[shots], targets,
                        directory / (label + "-generations.jsonl"), budget, reserve_seconds)}
            prior_metrics = summary["best_metrics"]
            if (abs(result["zero_shot"]["nll"] - prior_metrics["validation/loss"]) > 1e-4
                    or result["zero_shot"]["strict_correct"] != prior_metrics["validation/strict_correct"]):
                raise ValueError("Audit did not reproduce the saved checkpoint's original zero-shot validation.")
            result.update(status="completed", elapsed_seconds=time.monotonic() - started)
            save_object(directory / "result.json", result)
            results.append(result)
            print(json.dumps({"trial_id": identity, "zero_shot": result["zero_shot"],
                              "five_shot": result["five_shot"]}), flush=True)
            del replicated, parameters
    except BaseException as error:
        save_object(output / "interrupted.json", {"status": "interrupted",
            "error_type": type(error).__name__, "error": str(error),
            "completed_trials": len(results), "finished_unix_seconds": time.time()})
        raise
    save_object(output / "completed.json", {"status": "completed", "results": results,
        "protocol_sha256": digest, "test_used_for_selection": False, "finished_unix_seconds": time.time()})


if __name__ == "__main__":
    main()

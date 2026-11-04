"""Run a bounded, remotely guarded two-peer training and evaluation pilot."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any

from protocol import (Problem, State, boxed_integer, collect_tree,
                      question_fingerprint, sample_seed, transition, validate_settings)


def write_json(path: Path, value: Any) -> None:
    """Atomically publish a complete JSON record on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def file_record(path: Path) -> dict:
    """Record a file's exact bytes and content digest for durable artifact verification."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require_remote(receipt: dict, source: Path, maximum_compute_seconds: int = 86400) -> None:
    """Fail before data or model imports unless this is the explicitly audited remote VM."""
    if (socket.gethostname() != receipt["remote_hostname"] or socket.gethostname() == receipt["control_hostname"]
            or "kristine" not in receipt["tpu_name"].lower() or receipt["tpu_type"] != "v5litepod-8"
            or not receipt["spot"] or not str(source).startswith("/home/kristinelu/peer-tree/")
            or receipt["zone"].rsplit("-", 1)[0] != receipt["region"]):
        raise PermissionError("This program requires the exact authorized remote single-host Spot TPU.")
    if time.time() >= receipt["deadline_unix_seconds"]:
        raise TimeoutError("The persisted allocation budget has expired.")
    if not 0 < receipt["deadline_unix_seconds"] - receipt["allocation_created_unix_seconds"] <= maximum_compute_seconds <= 86400:
        raise ValueError("The allocation deadline exceeds the configured compute cap or absolute 24-hour maximum.")
    for relative_path, expected in receipt["source_sha256"].items():
        if file_record(source / relative_path)["sha256"] != expected:
            raise ValueError(f"Immutable source verification failed: {relative_path}")


def verify_checkpoint(directory: Path, settings: dict, receipt: dict) -> dict:
    """Reject incomplete, corrupted, incompatible, or budget-resetting resume inputs."""
    manifest = json.loads((directory / "complete.json").read_text())
    if (manifest["status"] != "complete" or manifest["model_revision"] != settings["model_revision"]
            or manifest["deadline_unix_seconds"] != receipt["deadline_unix_seconds"]
            or json.loads((directory / "configuration.json").read_text()) != settings):
        raise ValueError("Resume must preserve the completed checkpoint, configuration, and original deadline.")
    for record in manifest["files"]:
        path = directory / record["name"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint manifest path escapes its directory.")
        actual = file_record(path)
        if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
            raise ValueError(f"Checkpoint integrity failed: {record['name']}")
    return manifest


def prepare_data(settings: dict, output: Path, cache: Path, region: str) -> dict:
    """Use original competition questions and answer-only verification with disjoint splits."""
    from transformers import AutoTokenizer
    import random

    tokenizer = AutoTokenizer.from_pretrained(settings["model"], revision=settings["model_revision"], cache_dir=str(cache))
    if "frozen_data" in settings:
        frozen = settings["frozen_data"]
        if frozen["region"] != region:
            raise ValueError("Frozen split reuse requires the original data region.")
        directory = Path(frozen["directory"])
        fingerprints: set[str] = set()
        counts = {"reference_reasoning_exposed": False, "pretrained_contamination_audited": False,
                  "source_checkpoint": frozen["source_checkpoint"], "reused_frozen_splits": True,
                  "maximum_initial_prompt_tokens": 0, "files": frozen["files"]}
        for name, limit in (("train", "training_problem_limit"), ("validation", "validation_problem_limit"),
                            ("test", "evaluation_problem_limit")):
            path = directory / f"{name}.json"
            actual = file_record(path)
            if {key: actual[key] for key in ("bytes", "sha256")} != frozen["files"][path.name]:
                raise ValueError(f"Frozen split integrity failed: {path.name}")
            rows = json.loads(path.read_text())
            assert isinstance(rows, list)
            if len(rows) != settings[limit]:
                raise ValueError(f"Frozen split count differs from the configured limit: {name}")
            for row in rows:
                problem = Problem(**row)
                fingerprint = question_fingerprint(problem.problem)
                if problem.split != name or fingerprint in fingerprints:
                    raise ValueError("Frozen splits must retain their labels and be disjoint.")
                fingerprints.add(fingerprint)
                encoded = tokenizer.apply_chat_template(State(problem.problem).messages(), tokenize=True,
                                                         add_generation_prompt=True, enable_thinking=False)
                if len(encoded) > settings["maximum_prompt_tokens"]:
                    raise ValueError("A frozen question exceeds the new prompt budget; never silently filter it.")
                counts["maximum_initial_prompt_tokens"] = max(counts["maximum_initial_prompt_tokens"], len(encoded))
            counts[name] = len(rows)
        # Prompt-length filtering before the seeded shuffle would change the comparison's questions.
        for name in ("train", "validation", "test"):
            shutil.copyfile(directory / f"{name}.json", output / f"{name}.json")
        write_json(output / "data-audit.json", counts)
        return counts
    from datasets import load_dataset

    evaluation = load_dataset(settings["evaluation_dataset"], revision=settings["evaluation_dataset_revision"],
                              split="train", cache_dir=str(cache / "datasets"))
    test = [Problem(f"aime2024-{row['id']}", row["problem"], int(row["answer"]), "test") for row in evaluation]
    if len(test) != settings["evaluation_problem_limit"]:
        raise ValueError("AIME 2024 must contain exactly the frozen 30 questions.")
    forbidden = {question_fingerprint(row.problem) for row in test}
    seen = set(forbidden)
    candidates = []
    counts = {"source_rows": 0, "non_integer": 0, "overlap_or_duplicate": 0, "prompt_too_long": 0}
    for subject in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                    "number_theory", "prealgebra", "precalculus"):
        dataset = load_dataset(settings["training_dataset"], subject, revision=settings["training_dataset_revision"],
                               split="train", cache_dir=str(cache / "datasets"))
        for index, row in enumerate(dataset):
            counts["source_rows"] += 1
            answer = boxed_integer(row["solution"])
            if answer is None:
                counts["non_integer"] += 1
                continue
            fingerprint = question_fingerprint(row["problem"])
            if fingerprint in seen:
                counts["overlap_or_duplicate"] += 1
                continue
            seen.add(fingerprint)
            encoded = tokenizer.apply_chat_template(State(row["problem"]).messages(), tokenize=True,
                                                     add_generation_prompt=True, enable_thinking=False)
            # Reserve room for the sender's next complete message; never truncate the original problem.
            if len(encoded) > settings["maximum_prompt_tokens"] - settings["maximum_new_tokens"] - 64:
                counts["prompt_too_long"] += 1
                continue
            candidates.append(Problem(f"math-{subject}-{index}", row["problem"], answer, "train"))
    random.Random(settings["seed"]).shuffle(candidates)
    validation_count = settings["validation_problem_limit"]
    train_count = settings["training_problem_limit"]
    if len(candidates) < validation_count + train_count:
        raise ValueError("Not enough qualifying integer-answer MATH questions for the fixed split.")
    validation = [Problem(row.identifier, row.problem, row.answer, "validation") for row in candidates[:validation_count]]
    training = candidates[validation_count:validation_count + train_count]
    for name, rows in (("train", training), ("validation", validation), ("test", test)):
        write_json(output / f"{name}.json", [asdict(row) for row in rows])
    counts.update(eligible_integer_questions=len(candidates), train=len(training), validation=len(validation),
                  test=len(test), reference_reasoning_exposed=False,
                  pretrained_contamination_audited=False)
    write_json(output / "data-audit.json", counts)
    return counts


class Study:
    """Own the fixed deadline, append-only artifacts, W&B metrics, and one training lineage."""

    def __init__(self, settings: dict, receipt: dict, output: Path, resume: Path | None = None) -> None:
        self.settings = settings
        self.receipt = receipt
        self.output = output
        self.phase = "starting"
        self.wandb_run = None
        self.backend = None
        self.event_index = 0
        self.resume = resume
        self.completed_trees = 0

    def check_time(self, reserve: int = 0) -> None:
        """Stop before starting work that consumes the finalization reserve."""
        if time.time() + reserve >= self.receipt["deadline_unix_seconds"]:
            raise TimeoutError("Persisted pilot deadline reached.")

    def emit(self, event: str, values: dict) -> None:
        """Persist progress first and publish only scalar/configuration data to W&B."""
        record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **values}
        with (self.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        write_json(self.output / "status.json", {"status": "running", "phase": self.phase, **record})
        if self.wandb_run is not None:
            scalars = {f"{event}/{key}": value for key, value in values.items()
                       if isinstance(value, (int, float, bool))}
            self.wandb_run.log(scalars, step=self.event_index)
        self.event_index += 1

    def save_checkpoint(self, identity: str) -> dict:
        """Verify a checksummed adapter checkpoint and its same-region durable copy."""
        directory = self.output / "checkpoints" / identity
        self.backend.checkpoint(directory)
        evidence = directory / "evidence"
        evidence.mkdir()
        for path in sorted(self.output.iterdir()):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                shutil.copyfile(path, evidence / path.name)
        files = [{**file_record(path), "name": str(path.relative_to(directory))}
                 for path in sorted(directory.rglob("*")) if path.is_file()]
        manifest = {"status": "complete", "identity": identity,
                    "model_revision": self.settings["model_revision"],
                    "completed_trees": self.completed_trees,
                    "training_finished": identity in ("final", "completed"),
                    "deadline_unix_seconds": self.receipt["deadline_unix_seconds"],
                    "source_commit": self.receipt["source_commit"], "files": files}
        write_json(directory / "complete.json", manifest)
        verify_checkpoint(directory, self.settings, self.receipt)
        parent = self.receipt["gcs_prefix"] + "/attempts/" + self.output.name + "/checkpoints/"
        destination = parent + identity
        subprocess.run(["gcloud", "storage", "cp", "--recursive", str(directory),
                        parent], check=True)
        remote_manifest = json.loads(subprocess.check_output(["gcloud", "storage", "cat", destination + "/complete.json"]))
        if remote_manifest != manifest:
            raise ValueError("Durable checkpoint manifest differs from node-local checkpoint.")
        for record in manifest["files"]:
            if record["name"].startswith("evidence/"):
                continue  # gcloud cp verifies transport checksums; resume verifies every evidence SHA-256.
            size = int(subprocess.check_output(["gcloud", "storage", "objects", "describe",
                destination + "/" + record["name"], "--format=value(size)"]))
            if size != record["bytes"]:
                raise ValueError("Durable checkpoint size mismatch.")
        self.emit("checkpoint", {"identity": identity, "gcs": destination, "verified": True})
        if self.wandb_run is not None:
            self.wandb_run.summary["latest_checkpoint"] = destination
            self.wandb_run.summary["artifact_manifest"] = destination + "/complete.json"
        return manifest

    def evaluate(self, rows: list[Problem], label: str) -> dict:
        """Measure unbranched conversation pass@1 under fixed seeds without selecting on test results."""
        self.phase = label
        path = self.output / f"{label}.jsonl"
        if path.exists():
            raise FileExistsError("Evaluation artifacts are immutable; use a new attempt for a retry.")
        correct = 0
        total_tokens = 0
        reasons: dict[str, int] = {}
        for offset in range(0, len(rows), self.backend.batch_size):
            selected = rows[offset:offset + self.backend.batch_size]
            active = [{"problem": row, "state": State(row.problem), "events": [], "done": False} for row in selected]
            while any(not item["done"] for item in active):
                self.check_time(60)
                pending = [item for item in active if not item["done"]]
                agent = pending[0]["state"].agent
                generations = self.backend.generate([item["state"] for item in pending],
                    [sample_seed(self.settings["seed"], "evaluation", item["problem"].identifier,
                                 item["state"].turn) for item in pending], agent)
                for item, generation in zip(pending, generations, strict=True):
                    result = transition(item["state"], generation, self.settings["maximum_messages"])
                    total_tokens += len(generation.token_ids)
                    item["events"].append({"state": asdict(item["state"]), "generation": asdict(generation),
                                           "transition": asdict(result)})
                    if result.next_state is not None:
                        item["state"] = result.next_state
                    else:
                        reward = int(result.submission == item["problem"].answer) if result.submission is not None else 0
                        correct += reward
                        reasons[result.reason] = reasons.get(result.reason, 0) + 1
                        item.update(done=True, correct=reward, reason=result.reason)
            with path.open("a") as handle:
                for item in active:
                    handle.write(json.dumps({"problem": asdict(item["problem"]), "correct": item["correct"],
                                             "reason": item["reason"], "events": item["events"]}) + "\n")
            self.emit(label + "_progress", {"completed": min(offset + len(selected), len(rows)),
                                             "correct": correct, "generated_tokens": total_tokens})
        metrics = {"correct": correct, "total": len(rows), "pass_at_1": correct / len(rows),
                   "generated_tokens": total_tokens, "terminal_reasons": reasons}
        write_json(self.output / f"{label}-metrics.json", metrics)
        self.emit(label, metrics)
        return metrics

    def archive_failure(self) -> None:
        """Preserve failed-attempt evidence without accepting potentially partial optimizer state."""
        files = sorted(path for path in self.output.iterdir() if path.is_file() and path.suffix in (".json", ".jsonl"))
        manifest = {"source_commit": self.receipt["source_commit"], "files": [file_record(path) for path in files]}
        manifest_path = self.output / "failure-manifest.json"
        write_json(manifest_path, manifest)
        destination = self.receipt["gcs_prefix"] + "/attempts/" + self.output.name + "/failure-evidence/"
        subprocess.run(["gcloud", "storage", "cp", *map(str, files), str(manifest_path), destination], check=True)
        uploaded = json.loads(subprocess.check_output(["gcloud", "storage", "cat", destination + "failure-manifest.json"]))
        if uploaded != manifest:
            raise ValueError("Failed-attempt evidence manifest differs after upload.")
        self.wandb_run.summary["failure_evidence"] = destination

    def numerical_preflight(self) -> dict:
        """Compare the loaded TPU model with independent Hugging Face CPU logits on the remote VM."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM

        backend = self.backend
        torch.set_num_threads(8)
        reference = AutoModelForCausalLM.from_pretrained(str(backend.snapshot), torch_dtype=torch.float32,
                                                        attn_implementation="eager").eval()
        tokens = backend.tokenizer.encode("Compute 13 + 29. The result is", add_special_tokens=False)
        tensor = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            expected = reference(tensor).logits.float().numpy()
        weights = jax.tree.map(lambda array: array.addressable_shards[0].data.squeeze(0), backend.weights)
        adapters = jax.tree.map(lambda array: array.addressable_shards[0].data.squeeze(0), backend.states[0].params)

        @jax.jit
        def actual_forward(parameters: dict, adapter: tuple, input_ids: Any) -> Any:
            hidden, _ = backend.forward(parameters, adapter, input_ids, jnp.ones_like(input_ids),
                                         jnp.arange(input_ids.shape[1])[None, :])
            return hidden @ parameters["embedding"].T

        actual = np.asarray(actual_forward(weights, adapters, np.asarray([tokens], np.int32)))
        maximum_difference = float(np.max(np.abs(actual - expected)))
        if maximum_difference > 0.02 or int(actual[0, -1].argmax()) != int(expected[0, -1].argmax()):
            raise ValueError(f"Independent Qwen3 parity gate failed: maximum logit difference {maximum_difference}")
        del reference, expected, actual, weights, adapters
        import gc
        gc.collect()
        result = {"maximum_logit_difference": maximum_difference, "tolerance": 0.02,
                  "greedy_next_token_matches": True, "reference_device": "remote_cpu"}
        write_json(self.output / "numerical-preflight.json", result)
        self.emit("numerical_preflight", result)
        return result

    def single_agent_baseline(self, rows: list[Problem]) -> dict:
        """Evaluate the pretrained single agent with the same maximum total generated-token budget."""
        import jax
        from protocol import SINGLE_PROMPT
        self.phase = "single_agent_baseline"
        decoder = copy.copy(self.backend)
        decoder.output_width = self.settings["maximum_messages"] * self.settings["maximum_new_tokens"]
        decoder.total_width = decoder.prompt_width + decoder.output_width
        decoder.generate_parallel = jax.pmap(decoder.generate_device, devices=decoder.devices)
        correct = 0
        tokens = 0
        path = self.output / "single-agent-aime2024.jsonl"
        with path.open("x") as handle:
            for start in range(0, len(rows), decoder.batch_size):
                self.check_time(self.settings["finalization_reserve_seconds"])
                selected = rows[start:start + decoder.batch_size]
                messages = [[{"role": "system", "content": SINGLE_PROMPT},
                             {"role": "user", "content": row.problem}] for row in selected]
                generations = decoder.generate_messages(messages,
                    [sample_seed(self.settings["seed"], "single-agent", row.identifier) for row in selected], 0)
                for row, generation in zip(selected, generations, strict=True):
                    answer = boxed_integer(generation.text)
                    reward = int(answer == row.answer) if answer is not None else 0
                    correct += reward
                    tokens += len(generation.token_ids)
                    handle.write(json.dumps({"problem": asdict(row), "generation": asdict(generation),
                                             "correct": reward, "parsed_answer": answer}) + "\n")
                handle.flush()
                self.emit("single_agent_progress", {"completed": min(start + len(selected), len(rows)), "correct": correct})
        metrics = {"correct": correct, "total": len(rows), "pass_at_1": correct / len(rows),
                   "generated_tokens": tokens, "maximum_generated_tokens_per_problem": decoder.output_width,
                   "comparison": "equal maximum generated-token budget, actual compute may differ"}
        write_json(self.output / "single-agent-aime2024-metrics.json", metrics)
        self.emit("single_agent_baseline", metrics)
        return metrics

    def run(self) -> None:
        """Execute the fixed pilot and retain honest terminal status for every partial phase."""
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_DISABLE_GIT"] = "true"
        log_directory = self.output / "tpu-driver-logs"
        log_directory.mkdir()
        os.environ["TPU_LOG_DIR"] = str(log_directory)
        import wandb
        self.wandb_run = wandb.init(project=self.settings["wandb_project"], entity=self.settings["wandb_entity"],
            group=self.settings["wandb_group"], name=self.output.name, config={**self.settings, "execution_receipt": self.receipt},
            dir=str(self.output), settings=wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True))
        write_json(self.output / "wandb-run.json", {"id": self.wandb_run.id, "url": self.wandb_run.url})
        os.environ["JAX_PLATFORMS"] = "tpu,cpu"
        try:
            self.phase = "data_preparation"
            cache = self.output.parent / "cache"
            resume_manifest = None
            if self.resume is None:
                audit = prepare_data(self.settings, self.output, cache, self.receipt["region"])
                self.emit("data_audit", audit)
            else:
                resume_manifest = verify_checkpoint(self.resume, self.settings, self.receipt)
                self.completed_trees = resume_manifest["completed_trees"]
                for path in (self.resume / "evidence").iterdir():
                    if path.name not in ("execution-receipt.json", "effective-configuration.json", "wandb-run.json",
                                          "events.jsonl", "status.json", "terminal.json"):
                        shutil.copyfile(path, self.output / path.name)
                write_json(self.output / "resume-lineage.json", {"checkpoint": str(self.resume),
                    "manifest": resume_manifest, "previous_events": "preserved in source checkpoint evidence"})
                self.emit("resume", {"next_tree": self.completed_trees, "checkpoint": str(self.resume)})
            from qwen_backend import QwenBackend
            self.phase = "model_loading"
            self.backend = QwenBackend(self.settings, cache)
            self.emit("model_loaded", {"devices": len(self.backend.devices)})
            self.phase = "numerical_preflight"
            self.numerical_preflight()
            if self.resume is not None:
                self.backend.restore(self.resume)
            training = [Problem(**row) for row in json.loads((self.output / "train.json").read_text())]
            validation = [Problem(**row) for row in json.loads((self.output / "validation.json").read_text())]
            test = [Problem(**row) for row in json.loads((self.output / "test.json").read_text())]
            if self.resume is None:
                self.save_checkpoint("prepared")
            for rows, label in ((validation, "before_validation"), (test, "before_aime2024")):
                if not (self.output / f"{label}-metrics.json").exists():
                    self.evaluate(rows, label)
                    self.save_checkpoint(label)
            if not (self.output / "single-agent-aime2024-metrics.json").exists():
                self.single_agent_baseline(test)
            if self.resume is None or self.completed_trees == 0:
                self.save_checkpoint("initial")
            started = time.monotonic()
            stop_reason = "tree_limit"
            next_tree = self.completed_trees
            tree_limit = next_tree if resume_manifest and resume_manifest["training_finished"] else self.settings["maximum_training_trees"]
            for tree_index in range(next_tree, tree_limit):
                try:
                    self.check_time(self.settings["finalization_reserve_seconds"])
                    self.phase = "training_tree"
                    beginning = time.monotonic()
                    items, tree = collect_tree(self.backend, training[tree_index % len(training)], self.settings,
                        tree_index, lambda: self.check_time(self.settings["finalization_reserve_seconds"]))
                    write_json(self.output / f"tree-{tree_index:04d}.json", {**tree, "items": [asdict(item) for item in items]})
                    self.emit("tree_collected", {"tree": tree_index, "messages": len(items),
                        "terminal_success_rate": sum(tree["leaf_rewards"]) / len(tree["leaf_rewards"]),
                        "rollout_seconds": time.monotonic() - beginning,
                        "generated_tokens": sum(len(item.generation.token_ids) for item in items)})
                    if tree_index == next_tree:
                        self.emit("replay_gate", self.backend.verify_replay(items))
                    self.phase = "training_update"
                    # Finish the entire tree update atomically relative to its checkpoint; the hard supervisor still bounds it.
                    for metrics in self.backend.update(items, lambda: self.check_time(60)):
                        self.emit("update", {"tree": tree_index, **metrics})
                    self.completed_trees = tree_index + 1
                    self.save_checkpoint(f"tree-{tree_index:04d}")
                    if tree_index == next_tree:
                        memory = [device.memory_stats() for device in self.backend.devices]
                        peak = max(value["peak_bytes_in_use"] / value["bytes_limit"] for value in memory)
                        seconds = time.monotonic() - beginning
                        projected = seconds * (tree_limit - self.completed_trees)
                        self.emit("first_tree_gate", {"seconds": time.monotonic() - beginning,
                            "messages": len(items), "optimizer_updates_finite": True,
                            "checkpoint_verified": True, "maximum_peak_memory_fraction": peak,
                            "remaining_seconds_upper_projection": projected})
                        if peak > 0.9 or projected + time.time() + self.settings["finalization_reserve_seconds"] >= self.receipt["deadline_unix_seconds"]:
                            stop_reason = "memory_or_throughput_gate"
                            break
                except TimeoutError:
                    stop_reason = "finalization_reserve"
                    # Roll back any partial update rather than treating an incomplete tree as a completed policy step.
                    identity = "initial" if self.completed_trees == 0 else f"tree-{self.completed_trees - 1:04d}"
                    previous = self.output / "checkpoints" / identity
                    if not previous.exists():
                        previous = self.resume
                    if previous is not None:
                        self.backend.restore(previous)
                    break
            self.emit("training_finished", {"elapsed_seconds": time.monotonic() - started,
                                            "trees": self.completed_trees, "stop_reason": stop_reason})
            self.save_checkpoint("final")
            for rows, label in ((validation, "after_validation"), (test, "after_aime2024")):
                if not (self.output / f"{label}-metrics.json").exists():
                    self.evaluate(rows, label)
            self.phase = "completed"
            self.emit("completed", {"completed": True})
            write_json(self.output / "terminal.json", {"status": "completed", "time": time.time()})
            self.save_checkpoint("completed")
            self.wandb_run.finish()
        except BaseException as error:
            write_json(self.output / "terminal.json", {"status": "interrupted" if isinstance(error, TimeoutError) else "failed",
                "phase": self.phase, "error": repr(error), "time": time.time()})
            try:
                self.archive_failure()
            except Exception as archive_error:
                print(json.dumps({"failure_archive_error": repr(archive_error)}), flush=True)
            self.wandb_run.finish(exit_code=1)
            raise


def main() -> None:
    """Require a verified remote receipt before creating a new immutable run output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    source = Path(__file__).resolve().parent
    receipt = json.loads(arguments.receipt.read_text())
    settings = json.loads(arguments.config.read_text())
    require_remote(receipt, source, settings["maximum_compute_seconds"])
    validate_settings(settings)
    if not str(arguments.output.resolve()).startswith("/home/kristinelu/peer-tree/"):
        raise PermissionError("Outputs must remain in the dedicated node-local pilot directory.")
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_json(arguments.output / "effective-configuration.json", settings)
    write_json(arguments.output / "execution-receipt.json", receipt)
    Study(settings, receipt, arguments.output, arguments.resume).run()


if __name__ == "__main__":
    main()

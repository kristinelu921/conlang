"""Supervise authorized preflight/audit processes under an external wall-time deadline."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from audit_checkpoints import (read_object, save_object, verify_execution_receipt,
                               verify_source_manifest)


def supervise_process(process: subprocess.Popen[bytes], remaining_seconds: float) -> tuple[int, bool]:
    """Bound an owned subprocess group with TERM then KILL before its authorized deadline."""
    try:
        return process.wait(timeout=max(0.0, remaining_seconds - 10.0)), False
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            return process.wait(timeout=5.0), True
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return process.wait(timeout=5.0), True


def main() -> None:
    """Start only the selected remote phase with a persisted approval clock and process identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "audit"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    arguments = parser.parse_args()
    if not arguments.attempt_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                                       for character in arguments.attempt_id):
        raise ValueError("Attempt identity must contain only lowercase letters, digits and hyphens.")
    source = Path(__file__).resolve().parent
    protocol = read_object(arguments.protocol)
    receipt = read_object(arguments.execution_receipt)
    budget = verify_execution_receipt(receipt, protocol, source, socket.gethostname(), time.time())
    source_digest = verify_source_manifest(source)
    reserve = protocol["budget"]["finalization_reserve_seconds"]
    if budget.remaining(time.time()) - reserve <= 10:
        raise TimeoutError("Insufficient authorized time to start the requested process.")
    identity = arguments.phase + "-" + arguments.attempt_id
    control = source.parent / "control" / identity
    control.mkdir(parents=True, exist_ok=False)
    output = source.parent / "output" / identity
    command = [sys.executable, str(source / "audit_checkpoints.py"),
               "--protocol", str(arguments.protocol.resolve()),
               "--execution-receipt", str(arguments.execution_receipt.resolve()),
               "--output-directory", str(output)]
    if arguments.phase == "preflight":
        command.append("--preflight-only")
    environment = {**os.environ, "HF_HOME": "/home/kristinelu/smollm2-gsm8k/cache",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "USE_TORCH_XLA": "0", "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "tpu,cpu", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "PYTHONUNBUFFERED": "1", "JAX_COMPILATION_CACHE_DIR": "/home/kristinelu/smollm2-gsm8k/jax-cache"}
    with (control / "console.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=source, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            save_object(control / "started.json", {"status": "running", "pid": process.pid,
                "process_group": process.pid, "command": command, "hostname": socket.gethostname(),
                "source_manifest_sha256": source_digest,
                "execution_receipt_sha256": hashlib.sha256(arguments.execution_receipt.read_bytes()).hexdigest(),
                "study_started_unix_seconds": budget.started_unix_seconds,
                "authorized_seconds": budget.authorized_seconds,
                "phase_deadline_unix_seconds": budget.started_unix_seconds + budget.authorized_seconds - reserve})
            exit_code, timed_out = supervise_process(process, budget.remaining(time.time()) - reserve)
        except BaseException:
            supervise_process(process, 0.0)
            raise
    marker = output / ("preflight-passed.json" if arguments.phase == "preflight" else "completed.json")
    succeeded = exit_code == 0 and not timed_out and marker.exists()
    save_object(control / "finished.json", {"status": "completed" if succeeded else "interrupted" if timed_out else "failed",
        "exit_code": exit_code, "timed_out": timed_out, "completion_marker_present": marker.exists(),
        "finished_unix_seconds": time.time()})
    if not succeeded:
        raise SystemExit(124 if timed_out else exit_code or 1)


if __name__ == "__main__":
    main()

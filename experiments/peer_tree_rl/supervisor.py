"""Enforce a persisted allocation deadline on one exact child process group."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> None:
    """Start one immutable pilot attempt and terminate only its own child at the hard deadline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text())
    from run import require_remote
    require_remote(receipt, Path(__file__).resolve().parent)
    deadline = receipt["deadline_unix_seconds"]
    attempt = arguments.output.parent / (arguments.output.name + "-supervisor")
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).with_name("run.py")), "--config", str(arguments.config),
               "--receipt", str(arguments.receipt), "--output", str(arguments.output)]
    if arguments.resume is not None:
        command.extend(["--resume", str(arguments.resume)])
    with (attempt / "console.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (attempt / "identity.json").write_text(json.dumps({"pid": process.pid, "command": command,
            "supervisor_pid": os.getpid(), "deadline_unix_seconds": deadline}) + "\n")
        expired = False
        while process.poll() is None:
            if time.time() >= deadline:
                expired = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(min(10, max(0.1, deadline - time.time())))
        return_code = process.wait()
    (attempt / "terminal.json").write_text(json.dumps({"exit_code": return_code,
        "deadline_expired": expired, "finished_unix_seconds": time.time()}) + "\n")
    raise SystemExit(return_code if return_code >= 0 else 1)


if __name__ == "__main__":
    main()

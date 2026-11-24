"""Enforce the single approved deadline for the immutable remote training process group."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
sys.path.insert(0, str(ROOT / 'source'))
from audit_checkpoints import read_object, save_object, verify_execution_receipt, verify_source_manifest
from run_remote_phase import supervise_process


def main() -> None:
    """Launch one fresh attempt after remote tests and persist its exact terminal status."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id', required=True)
    arguments = parser.parse_args()
    identity = arguments.attempt_id
    if not identity or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in identity):
        raise ValueError('Invalid attempt identity.')
    protocol = read_object(ROOT / 'protocol.json')
    receipt = read_object(ROOT / 'execution-receipt.json')
    budget = verify_execution_receipt(receipt, protocol, ROOT / 'source', socket.gethostname(), time.time())
    verify_source_manifest(ROOT / 'source')
    digest = verify_source_manifest(SOURCE)
    if not (SOURCE / 'remote-tests-passed.json').exists():
        raise ValueError('Focused remote tests must pass before training.')
    if read_object(SOURCE / 'remote-tests-passed.json')['source_manifest_sha256'] != digest:
        raise ValueError('Remote tests covered a different source snapshot.')
    control = ROOT / 'control' / ('search-' + identity)
    control.mkdir(exist_ok=False)
    command = [sys.executable, str(SOURCE / 'matched_train.py'), '--attempt-id', identity]
    environment = {**os.environ, 'HF_HOME': '/home/kristinelu/smollm2-gsm8k/cache',
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'USE_TORCH_XLA': '0', 'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'JAX_PLATFORMS': 'tpu,cpu', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'PYTHONUNBUFFERED': '1', 'CLOUDSDK_PYTHON': '/usr/bin/python3',
        'JAX_COMPILATION_CACHE_DIR': '/home/kristinelu/smollm2-gsm8k/jax-cache'}
    with (control / 'console.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            save_object(control / 'started.json', {'status': 'running', 'pid': process.pid,
                'process_group': process.pid, 'command': command, 'source_manifest_sha256': digest,
                'deadline_unix_seconds': budget.started_unix_seconds + budget.authorized_seconds,
                'started_unix_seconds': time.time()})
            exit_code, timed_out = supervise_process(process, budget.remaining(time.time()))
        except BaseException:
            supervise_process(process, 0)
            raise
    complete = ROOT / 'output' / ('search-' + identity) / 'study-completed.json'
    status = 'completed' if exit_code == 0 and complete.exists() else 'interrupted' if timed_out else 'failed'
    result = {'status': status, 'exit_code': exit_code, 'timed_out': timed_out,
        'attempt_id': identity, 'completion_marker_present': complete.exists(), 'finished_unix_seconds': time.time()}
    save_object(control / 'finished.json', result)
    if status == 'completed' or timed_out:
        save_object(ROOT / 'study-terminal.json', result)
    raise SystemExit(exit_code if exit_code else 0 if status == 'completed' else 1)


if __name__ == '__main__':
    main()

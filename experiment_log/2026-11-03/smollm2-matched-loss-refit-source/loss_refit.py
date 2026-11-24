"""Run the optional locked loss recipe sequentially within the original study deadline."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

ROOT = Path('/home/kristinelu/smollm2-gsm8k/matched-validation-20261103')
SOURCE = Path(__file__).resolve().parent
TRAINING_SOURCE = ROOT / 'training-source-v3'
SCOPE = ROOT / 'loss-refit-control'
OUTPUT = ROOT / 'output/search-loss-final'
LOCK_SHA256 = '56423de9991cb1edf8558d5fc987216cbb1a8ddab5e5b6ee1ea219c5b04918f7'
TRAINING_SHA256 = 'ad401734ecda702bde838ce8e9d8625b2c07461862b650cf11fa8eec6ba0b8da'
sys.path.insert(0, str(ROOT / 'source'))
from audit_checkpoints import read_object, save_object, verify_execution_receipt, verify_source_manifest
from run_remote_phase import supervise_process
from matched_protocol import ComputeBudget
from loss_policy import loss_recipe, required_seconds, require_primary_finished, training_lock


def verified_inputs() -> tuple[dict[str, Any], ComputeBudget, dict[str, Any]]:
    """Verify the authorized host, immutable source, original deadline and pre-test recipe lock."""
    if SOURCE != ROOT / 'loss-refit-source-v1':
        raise PermissionError('Use the dedicated TPU node-local source snapshot.')
    protocol = read_object(ROOT / 'protocol.json')
    budget = verify_execution_receipt(read_object(ROOT / 'execution-receipt.json'),
        {**protocol, 'budget': {**protocol['budget'], 'finalization_reserve_seconds': 0}},
        ROOT / 'source', socket.gethostname(), time.time())
    digest = verify_source_manifest(SOURCE)
    if read_object(SOURCE / 'remote-tests-passed.json')['source_manifest_sha256'] != digest:
        raise ValueError('Remote tests did not cover this source closure.')
    if verify_source_manifest(TRAINING_SOURCE) != TRAINING_SHA256:
        raise ValueError('The shared tested trainer changed.')
    lock_path = ROOT / 'output/search-r1/locked-selection.json'
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError('The original pre-test selection lock changed.')
    primary_pid = read_object(ROOT / 'control/search-r1/started.json')['pid']
    require_primary_finished(read_object(ROOT / 'control/search-r1/finished.json'),
        read_object(ROOT / 'wandb-completed.json'), Path(f'/proc/{primary_pid}').exists())
    return protocol, budget, read_object(lock_path)


def train() -> None:
    """Reuse the tested trainer and exporter with a distinct loss-refit checkpoint namespace."""
    protocol, budget, selection = verified_inputs()
    sys.path.insert(0, str(TRAINING_SOURCE))
    from matched_train import MatchedStudy, finalize_model
    recipe = loss_recipe(selection)
    study = MatchedStudy(protocol, budget, OUTPUT)
    study.reserve = 900
    study.configuration['artifact_bucket_prefix'] = (
        'gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-matched-validation-20261103/loss-refit/checkpoints')
    save_object(OUTPUT / 'loss-effective-configuration.json', {
        **read_object(OUTPUT / 'effective-configuration.json'),
        'engine_configuration': study.configuration,
        'wrapper_manifest_sha256': verify_source_manifest(SOURCE),
        'parent_selection_sha256': LOCK_SHA256,
        'amendment': 'Use an isolated GCS prefix; retain the exact tested training and export implementation.'})
    final = study.train_trial(recipe, recipe['epochs'], final_fit=True)
    if final['completed_epochs'] != recipe['epochs']:
        raise TimeoutError('The locked loss refit did not finish all epochs.')
    save_object(OUTPUT / 'refit-completed.json', {'status': 'completed',
        'selection': read_object(OUTPUT / 'locked-selection.json'), 'final': final,
        'finished_unix_seconds': time.time()})
    finalize_model(study, final)


def supervise() -> None:
    """Admit a sequential continuation and supervise both training and scalar publication."""
    protocol, budget, selection = verified_inputs()
    primary = ROOT / 'output/search-r1'
    refit = read_object(primary / 'refit-completed.json')
    completion = read_object(primary / 'study-completed.json')
    durations = [row['metrics']['elapsed_epoch_seconds'] for row in refit['final']['epochs']]
    durations += [row['metrics']['elapsed_epoch_seconds'] for row in
        read_object(primary / 'matched-02-r1/summary.json')['epochs']]
    recipe = loss_recipe(selection)
    estimate = required_seconds(recipe['epochs'], durations,
        completion['finished_unix_seconds'] - refit['finished_unix_seconds'])
    if budget.remaining(time.time()) < estimate:
        raise PermissionError('Insufficient original budget for the optional loss refit and verification.')
    SCOPE.mkdir(exist_ok=False)
    (SCOPE / 'source').symlink_to(ROOT / 'source', target_is_directory=True)
    (SCOPE / 'output').mkdir()
    OUTPUT.mkdir(exist_ok=False)
    (SCOPE / 'output/search-loss-final').symlink_to(OUTPUT, target_is_directory=True)
    control = SCOPE / 'control'
    control.mkdir()
    save_object(control / 'admission.json', {'remaining_seconds': budget.remaining(time.time()),
        'required_seconds': estimate, 'measured_epoch_seconds': durations,
        'primary_finalization_seconds': completion['finished_unix_seconds'] - refit['finished_unix_seconds'],
        'parent_selection_sha256': LOCK_SHA256, 'recipe': recipe,
        'deadline_unix_seconds': budget.started_unix_seconds + budget.authorized_seconds})
    save_object(OUTPUT / 'locked-selection.json', {**selection,
        'selected': selection['minimum_zero_shot_nll'], 'refit_recipe': recipe,
        'primary_balanced_selection': selection['selected'], 'selection_objective': 'minimum_zero_shot_validation_nll',
        'parent_selection_sha256': LOCK_SHA256, 'secondary_launch_unix_seconds': time.time()})
    save_object(OUTPUT / 'frozen-reference.json', read_object(primary / 'frozen-reference.json'))
    environment = {**os.environ, 'HF_HOME': '/home/kristinelu/smollm2-gsm8k/cache',
        'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
        'USE_TORCH_XLA': '0', 'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '',
        'JAX_PLATFORMS': 'tpu,cpu', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'PYTHONUNBUFFERED': '1', 'CLOUDSDK_PYTHON': '/usr/bin/python3',
        'JAX_COMPILATION_CACHE_DIR': '/home/kristinelu/smollm2-gsm8k/jax-cache'}
    command = [sys.executable, str(SOURCE / 'loss_refit.py'), '--phase', 'train']
    publisher_command = ['/home/kristinelu/smollm2-gsm8k/wandb-runtime/bin/python', str(SOURCE / 'wandb_bridge.py')]
    with (control / 'console.log').open('xb') as log, (control / 'wandb.log').open('xb') as publisher_log:
        publisher = subprocess.Popen(publisher_command, cwd=SOURCE,
            env={**environment, 'JAX_PLATFORMS': 'cpu'}, stdout=publisher_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        save_object(control / 'started.json', {'pid': process.pid, 'publisher_pid': publisher.pid,
            'command': command, 'publisher_command': publisher_command,
            'source_manifest_sha256': verify_source_manifest(SOURCE), 'parent_selection_sha256': LOCK_SHA256,
            'started_unix_seconds': time.time(), 'deadline_unix_seconds': budget.started_unix_seconds + budget.authorized_seconds})
        try:
            exit_code, timed_out = supervise_process(process, budget.remaining(time.time()) - 180.0)
        except BaseException:
            supervise_process(process, 0)
            supervise_process(publisher, 0)
            raise
        status = 'completed' if exit_code == 0 and (OUTPUT / 'study-completed.json').exists() else 'failed'
        result = {'status': status, 'exit_code': exit_code, 'timed_out': timed_out,
            'finished_unix_seconds': time.time()}
        save_object(SCOPE / 'study-terminal.json', result)
        publisher_code, publisher_timed_out = supervise_process(publisher, min(180.0, budget.remaining(time.time())))
        save_object(control / 'finished.json', {**result, 'publisher_exit_code': publisher_code,
            'publisher_timed_out': publisher_timed_out, 'publisher_completion_present': (SCOPE / 'wandb-completed.json').exists()})
    if status != 'completed' or publisher_code != 0 or not (SCOPE / 'wandb-completed.json').exists():
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('supervise', 'train'), required=True)
    arguments = parser.parse_args()
    if arguments.phase == 'train':
        with training_lock(SCOPE / 'accelerator.lock'):
            train()
    else:
        supervise()

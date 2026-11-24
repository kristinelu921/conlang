"""Stream scalar matched-study metrics to W&B from a separate CPU-only remote process."""
from __future__ import annotations
import hashlib
import json
import netrc
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path('/home/kristinelu/smollm2-gsm8k/matched-validation-20261103')
sys.path.insert(0, str(ROOT / 'source'))
from audit_checkpoints import read_object, save_object


def main() -> None:
    """Publish only study configuration and scalar metrics without code, samples or model artifacts."""
    authentication = netrc.netrc('/home/kristinelu/smollm2-gsm8k/sft-control-20261102/.wandb.netrc').authenticators('api.wandb.ai')
    assert authentication is not None
    os.environ['WANDB_API_KEY'] = authentication[2]
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['USE_TORCH_XLA'] = '0'
    import wandb
    fields = ('learning_rate', 'batch_size', 'epochs', 'schedule_epochs', 'minimum_epochs', 'patience',
        'seed', 'warmup_fraction', 'final_learning_rate_ratio', 'weight_decay', 'five_shot_training_fraction', 'final_fit', 'label_smoothing', 'final_answer_weight')
    common = {'model': 'HuggingFaceTB/SmolLM2-135M', 'dataset': 'openai/gsm8k', 'precision': 'float32',
        'split_seed': 20261102, 'validation_examples': 747, 'hardware': 'dedicated Spot v5e-8',
        'test_used_for_selection': False}
    settings = wandb.Settings(x_disable_stats=True, disable_git=True, disable_code=True, quiet=True)
    active: dict[str, Any] = {}
    positions: dict[str, int] = {}
    finished: set[str] = set()
    audit_finished: set[str] = set()
    while True:
        for result_path in sorted((ROOT / 'output').glob('audit-*/*/result.json')):
            identity = result_path.parent.name
            if identity in audit_finished or (result_path.parent / 'wandb-run.json').exists():
                audit_finished.add(identity)
                continue
            result = read_object(result_path)
            run = wandb.init(project='conlangs', entity='kaynelu921-massachusetts-institute-of-technology',
                group='smollm2-gsm8k-matched-validation-20261103', name='matched-audit-' + identity,
                job_type='checkpoint-validation', config={**common, 'checkpoint_trial': identity, 'epoch': result['epoch'],
                    'protocol_sha256': result['protocol_sha256']}, dir=str(ROOT), reinit='create_new', settings=settings)
            metrics = {'epoch': result['epoch']}
            for condition in ('zero_shot', 'five_shot'):
                metrics.update({'validation/' + condition + '/' + key: value for key, value in result[condition].items()})
            run.log(metrics)
            run.summary.update(metrics)
            record = {'id': run.id, 'url': run.url, 'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest()}
            run.finish()
            save_object(result_path.parent / 'wandb-run.json', record)
            print(json.dumps(record), flush=True)
            audit_finished.add(identity)
        for events_path in sorted((ROOT / 'output').glob('search-*/*/events.jsonl')):
            directory = events_path.parent
            identity = directory.parent.name + '/' + directory.name
            if identity in finished:
                continue
            if identity not in active:
                if (directory / 'wandb-run.json').exists():
                    raise RuntimeError('A previous W&B bridge exists; reconcile before any duplicate publication.')
                configuration = read_object(directory / 'config.json')
                run = wandb.init(project='conlangs', entity='kaynelu921-massachusetts-institute-of-technology',
                    group='smollm2-gsm8k-matched-validation-20261103', name='gsm8k-' + directory.name,
                    job_type='final-refit' if configuration['final_fit'] else 'finetune-search',
                    config={**common, **{field: configuration[field] for field in fields}},
                    dir=str(ROOT), reinit='create_new', settings=settings)
                run.define_metric('epoch')
                run.define_metric('*', step_metric='epoch')
                save_object(directory / 'wandb-run.json', {'id': run.id, 'url': run.url})
                print(json.dumps({'trial': identity, 'url': run.url}), flush=True)
                active[identity] = run
                positions[identity] = 0
            run = active[identity]
            with events_path.open() as handle:
                handle.seek(positions[identity])
                while True:
                    previous = handle.tell()
                    line = handle.readline()
                    if not line.endswith('\n'):
                        handle.seek(previous)
                        break
                    event = json.loads(line)
                    if event['event'] == 'metrics':
                        run.log({key: value for key, value in event['metrics'].items() if isinstance(value, (float, int))})
                    elif event['event'] == 'completed':
                        run.summary.update(event['summary'])
                        run.finish()
                        finished.add(identity)
                        del active[identity]
                        break
                positions[identity] = handle.tell()
        if (ROOT / 'study-terminal.json').exists():
            terminal = read_object(ROOT / 'study-terminal.json')
            for identity, run in active.items():
                run.summary['study_terminal_status'] = terminal['status']
                run.finish(exit_code=1)
            completions = list((ROOT / 'output').glob('search-*/study-completed.json'))
            for completion_path in completions:
                completion = read_object(completion_path)
                run = wandb.init(project='conlangs', entity='kaynelu921-massachusetts-institute-of-technology',
                    group='smollm2-gsm8k-matched-validation-20261103', name='gsm8k-matched-final-benchmark',
                    job_type='final-benchmark', config=common, dir=str(ROOT), reinit='create_new', settings=settings)
                metrics = {'train/loss': completion['final_train_loss'], 'test/loss': completion['final_test_loss']}
                for shots, values in completion['benchmarks'].items():
                    metrics.update({'test/' + shots + 'shot/' + key: value for key, value in values.items()
                                    if isinstance(value, (float, int))})
                run.log(metrics)
                run.summary.update(metrics)
                record = {'id': run.id, 'url': run.url}
                run.finish()
                save_object(completion_path.parent / 'benchmark-wandb-run.json', record)
            save_object(ROOT / 'wandb-completed.json', {'status': 'completed', 'trials': sorted(finished),
                'audits': sorted(audit_finished), 'study_terminal_status': terminal['status'],
                'finished_unix_seconds': time.time()})
            break
        time.sleep(5)


if __name__ == '__main__':
    main()

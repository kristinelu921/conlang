"""Train and select matched zero/five-shot GSM8K candidates on the authorized remote TPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parent
sys.path.insert(0, str(ROOT / 'source'))
from audit_checkpoints import (read_object, save_object, verify_execution_receipt, verify_source_manifest,
                               evaluate_loss, evaluate_generations, require_search_time)
from matched_protocol import (Question, PromptExample, ComputeBudget, ValidationResult, demonstration_pool,
    prepare_prompt, protocol_digest, validate_partition, mixed_query_indices, balanced_selection_key)
from training_policy import should_stop, nondominated, epoch_admission_seconds


class MatchedStudy:
    """Own one model, immutable validation prompts and sequential training checkpoints."""

    def __init__(self, protocol: dict[str, Any], budget: ComputeBudget, output: Path) -> None:
        sys.path.insert(0, str(ROOT / 'source' / 'legacy'))
        from research import StudyEngine
        self.protocol = protocol
        self.budget = budget
        self.reserve = protocol['budget']['finalization_reserve_seconds']
        self.output = output
        old_root = Path('/home/kristinelu/smollm2-gsm8k/research-20261102')
        self.configuration = {**read_object(old_root / 'study_config.json'), 'output_directory': str(output),
            'artifact_bucket_prefix': 'gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-matched-validation-20261103/checkpoints'}
        self.engine = StudyEngine(self.configuration, output)
        split = read_object(output / 'split.json')
        if split != read_object(old_root / 'output' / 'split.json'):
            raise ValueError('Optimization partition changed.')
        self.training = [Question(index, **self.engine.rows[index]) for index in split['train_indices']]
        self.validation = [Question(index, **self.engine.rows[index]) for index in split['validation_indices']]
        validate_partition(self.training, self.validation, len(self.engine.rows))
        self.pool = demonstration_pool(self.training)
        self.prompts = {shots: [self.prompt(row, shots, 0, validation=True) for row in self.validation]
                        for shots in (0, 5)}
        self.targets = [row.answer for row in self.validation]
        self.digest = protocol_digest(self.prompts[0], self.prompts[5], self.targets)
        parity = read_object(ROOT / 'output/preflight-first/preflight-passed.json')
        if parity['protocol_sha256'] != self.digest:
            raise ValueError('Numerical preflight used different validation prompts.')
        self.records: list[dict[str, Any]] = []
        self.epoch_durations: dict[float, list[float]] = {}
        self.reference: ValidationResult | None = None
        self.reference_record: dict[str, Any] | None = None
        self.selected: dict[str, Any] | None = None
        self.minimum_loss: dict[str, Any] | None = None
        self.scratch = Path('/dev/shm') / ('smollm2-matched-' + str(int(budget.started_unix_seconds)) + '-' + output.name)
        self.scratch.mkdir(exist_ok=False)
        save_object(output / 'effective-configuration.json', {**protocol, 'engine_configuration': self.configuration,
            'source_manifest_sha256': verify_source_manifest(SOURCE), 'validation_protocol_sha256': self.digest,
            'checkpoint_staging': str(self.scratch), 'checkpoint_staging_is_transient': True,
            'checkpoint_durability': 'same-region GCS upload and completion marker verified before epoch acceptance'})

    def prompt(self, question: Question, shots: int, epoch: int, validation: bool = False) -> PromptExample:
        """Construct a complete prompt under the fixed leakage-controlled protocol."""
        settings = self.protocol['prompt_protocol']
        return prepare_prompt(question, self.pool, self.engine.backend.tokenizer, shots,
            settings['validation_demonstration_seed' if validation else 'training_demonstration_seed'], epoch,
            self.engine.backend.config.max_position_embeddings, settings['answer_token_reserve'],
            settings['maximum_demonstration_sampling_attempts'])

    def save_checkpoint(self, state: Any, identity: str, epoch: int, metadata: dict[str, Any]) -> dict[str, Any]:
        """Mirror a complete optimizer state from node-local RAM storage without deleting old checkpoints."""
        import jax
        import numpy as np
        from flax import serialization
        from train_gsm8k import file_record, mirror_directory
        require_search_time(self.budget, 60)
        if shutil.disk_usage(self.scratch).free < 64 * 1024**3:
            raise OSError('Checkpoint RAM staging must preserve 64 GiB free.')
        directory = self.scratch / f'{identity}-epoch{epoch:02d}'
        directory.mkdir()
        host_state = jax.tree.map(lambda value: np.asarray(value[0]), state)
        path = directory / 'training_state.msgpack'
        with path.open('xb') as handle:
            handle.write(serialization.to_bytes(host_state))
            handle.flush()
            os.fsync(handle.fileno())
        save_object(directory / 'metadata.json', metadata)
        manifest = {'status': 'complete', 'step': int(host_state.step),
            'files': [file_record(path), file_record(directory / 'metadata.json')]}
        save_object(directory / 'complete.json', manifest)
        mirror_directory(directory, self.configuration['artifact_bucket_prefix'])
        destination = self.configuration['artifact_bucket_prefix'] + '/' + directory.name
        remote_manifest = json.loads(subprocess.check_output(['gcloud', 'storage', 'cat', destination + '/complete.json']))
        if remote_manifest != manifest:
            raise ValueError('Checkpoint completion marker differs after upload.')
        remote_size = int(subprocess.check_output(['gcloud', 'storage', 'objects', 'describe',
            destination + '/training_state.msgpack', '--format=value(size)']).strip())
        if remote_size != manifest['files'][0]['bytes']:
            raise ValueError('Durable checkpoint object size differs.')
        result = {'local_staging_directory': str(directory), 'gcs_directory': destination,
                  'manifest': manifest, 'durable_size_verified': True}
        del host_state
        return result

    def evaluate(self, parameters: Any, directory: Path, identity: str, epoch: int) -> tuple[ValidationResult, dict[str, Any]]:
        """Measure both fixed validation formats against all 747 held-out queries."""
        conditions = {}
        for shots, label in ((0, 'zero_shot'), (5, 'five_shot')):
            conditions[label] = {**evaluate_loss(self.engine, parameters, self.prompts[shots], self.budget, self.reserve),
                **evaluate_generations(self.engine, parameters, self.prompts[shots], self.targets,
                    directory / f'{label}-epoch{epoch:02d}.jsonl', self.budget, self.reserve)}
        result = ValidationResult(identity, epoch, len(self.validation),
            conditions['zero_shot']['strict_correct'], conditions['five_shot']['strict_correct'],
            conditions['zero_shot']['nll'], conditions['five_shot']['nll'], self.digest)
        return result, conditions

    def train_trial(self, specification: dict[str, Any], epochs: int, final_fit: bool = False) -> dict[str, Any]:
        """Train a base-initialized recipe with exact query coverage and retained epoch states."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        import optax
        from flax.training.train_state import TrainState
        from train_gsm8k import make_batch, tied_parameters, encode_example
        from smoothing import smoothed_loss_sum
        from gradient_accumulation import accumulated_gradients
        from research import weighted_example
        settings = {**self.protocol['candidate_defaults'], **specification, 'epochs': epochs, 'final_fit': final_fit}
        settings['gradient_accumulation'] = 4 if settings['five_shot_training_fraction'] else 1
        settings['microbatch_size'] = settings['batch_size'] // settings['gradient_accumulation']
        if self.engine.backend.config.attention_dropout != 0.0:
            raise ValueError('Exact accumulation requires the pinned zero-dropout model.')
        identity = settings['id']
        directory = self.output / identity
        directory.mkdir()
        save_object(directory / 'config.json', settings)
        training = [Question(index, **row) for index, row in enumerate(self.engine.rows)] if final_fit else self.training
        if final_fit:
            # Final refit has a locked recipe; all original train examples may now supply demonstrations.
            self.pool = demonstration_pool(training)
        batch_size = settings['batch_size']
        steps_per_epoch = math.ceil(len(training) / batch_size)
        total_steps = steps_per_epoch * settings['schedule_epochs']
        schedule = optax.warmup_cosine_decay_schedule(init_value=0., peak_value=settings['learning_rate'],
            warmup_steps=max(1, math.ceil(total_steps * settings['warmup_fraction'])), decay_steps=total_steps,
            end_value=settings['learning_rate'] * settings['final_learning_rate_ratio'])
        optimizer = optax.chain(optax.clip_by_global_norm(settings['gradient_clip_norm']),
            optax.adamw(schedule, b1=settings['adam_beta1'], b2=settings['adam_beta2'],
                eps=settings['adam_epsilon'], weight_decay=settings['weight_decay']))
        with jax.default_device(jax.devices('cpu')[0]):
            host_state = TrainState.create(apply_fn=self.engine.backend.model.__call__,
                params=self.engine.original_parameters, tx=optimizer)
        state = jax.device_put_replicated(host_state, self.engine.backend.devices)
        del host_state

        def training_step(current: TrainState, batch: dict[str, Any]) -> tuple[TrainState, dict[str, Any]]:
            """Apply the globally token-weighted query-answer gradient through tied embeddings."""
            count = jax.lax.psum(jnp.sum(batch['loss_mask'][:, 1:]), 'devices')
            def objective(parameters: dict[str, Any], microbatch: dict[str, Any]) -> tuple[Any, Any]:
                """Return the locally summed loss with the global supervised-token denominator."""
                random_key = jax.random.fold_in(jax.random.PRNGKey(settings['seed']), current.step)
                random_key = jax.random.fold_in(random_key, jax.lax.axis_index('devices'))
                logits = self.engine.backend.model(microbatch['input_ids'], attention_mask=microbatch['attention_mask'],
                    params=tied_parameters(parameters), train=True, dropout_rng=random_key).logits
                numerator, _ = smoothed_loss_sum(logits, microbatch['input_ids'], microbatch['loss_mask'], settings['label_smoothing'])
                return numerator / count, numerator
            numerator, gradients = accumulated_gradients(current.params, batch, objective, settings['gradient_accumulation'])
            gradients = jax.lax.psum(gradients, 'devices')
            return current.apply_gradients(grads=gradients), {'train/update_loss': jax.lax.psum(numerator, 'devices') / count,
                'train/gradient_norm': optax.global_norm(gradients), 'train/learning_rate': schedule(current.step),
                'train/answer_tokens': count}

        parallel_step = jax.pmap(training_step, axis_name='devices', donate_argnums=(0,), devices=self.engine.backend.devices)
        test_examples = None
        if final_fit:
            from datasets import load_dataset
            if not (self.output / 'locked-selection.json').exists():
                raise PermissionError('Test access requires an immutable selection lock.')
            test_rows = load_dataset(self.protocol['dataset'], 'main', revision=self.protocol['dataset_revision'], split='test')
            test_examples = [encode_example(row['question'], row['answer'], self.engine.backend.tokenizer,
                self.engine.backend.config.max_position_embeddings) for row in test_rows]
        evaluation_examples = self.engine.examples if final_fit else self.engine.training_examples
        durations: list[float] = []
        trial_records = []
        best_key = None
        best_loss = math.inf
        last_improvement = 0
        with (directory / 'events.jsonl').open('x') as events:
            def emit(metrics: dict[str, Any]) -> None:
                """Persist scalar metrics for the isolated W&B bridge."""
                events.write(json.dumps({'event': 'metrics', 'metrics': metrics}, allow_nan=False) + '\n')
                events.flush()
                if 'validation/zero_shot/nll' in metrics or int(metrics.get('step', 0)) % 100 == 0:
                    print(json.dumps({'trial_id': identity, **metrics}, allow_nan=False), flush=True)
            initial = {'epoch': 0, 'step': 0, 'train/loss': self.engine.loss(state.params, evaluation_examples)['loss']}
            if test_examples is not None:
                initial['test/loss'] = self.engine.loss(state.params, test_examples)['loss']
            emit(initial)
            for epoch in range(1, epochs + 1):
                reserve = 900 if final_fit else self.reserve
                if not self.budget.admits(time.time(), epoch_admission_seconds(durations), reserve):
                    break
                started = time.monotonic()
                mixed = mixed_query_indices([row.source_index for row in training], settings['five_shot_training_fraction'],
                    self.protocol['prompt_protocol']['training_demonstration_seed'], epoch - 1)
                examples = [self.prompt(row, 5 if row.source_index in mixed else 0, epoch - 1) for row in training]
                encoded = [(list(example.input_tokens), list(example.loss_mask)) for example in examples]
                if settings['final_answer_weight'] != 1.0:
                    if mixed:
                        raise ValueError('Historical answer weighting is supported only for its zero-shot recipe.')
                    encoded = [weighted_example({'question': row.question, 'answer': row.answer}, self.engine.backend,
                        640, settings['final_answer_weight']) for row in training]
                width = max(640, math.ceil(max(len(example.input_tokens) for example in examples) / 128) * 128)
                save_object(directory / f'coverage-epoch{epoch:02d}.json', {'epoch': epoch,
                    'source_indices': [example.query_index for example in examples], 'five_shot_queries': sorted(mixed),
                    'demonstrations': [example.demonstration_indices for example in examples], 'sequence_width': width,
                    'prompt_tokens_sha256': hashlib.sha256(json.dumps([example.prompt_tokens for example in examples]).encode()).hexdigest()})
                order = np.random.default_rng(settings['seed'] + epoch - 1).permutation(len(examples))
                training_started = time.monotonic()
                steady_durations = []
                for batch_number, offset in enumerate(range(0, len(examples), batch_size)):
                    require_search_time(self.budget, reserve)
                    batch = make_batch(encoded, order[offset:offset + batch_size], batch_size, width,
                                       self.engine.backend.tokenizer.pad_token_id)
                    batch = {key: value.reshape(8, batch_size // 8, -1) for key, value in batch.items()}
                    step_started = time.monotonic()
                    state, device_metrics = parallel_step(state, batch)
                    metrics = {key: float(np.asarray(value[0])) for key, value in device_metrics.items()}
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError('Nonfinite training metric.')
                    if batch_number >= 3:
                        steady_durations.append(time.monotonic() - step_started)
                    step = (epoch - 1) * steps_per_epoch + batch_number + 1
                    metrics.update(step=step, epoch=epoch - 1 + (batch_number + 1) / steps_per_epoch)
                    emit(metrics)
                    if batch_number == 2:
                        print(json.dumps({'trial_id': identity, 'epoch': epoch, 'event': 'memory',
                            'devices': [device.memory_stats() for device in self.engine.backend.devices]}), flush=True)
                metrics = {'epoch': epoch, 'step': step, 'train/loss': self.engine.loss(state.params, evaluation_examples)['loss'],
                    'train/epoch_training_seconds': time.monotonic() - training_started, 'train/sequence_width': width,
                    'train/steady_examples_per_second': batch_size / (sum(steady_durations) / len(steady_durations))}
                if test_examples is not None:
                    metrics['test/loss'] = self.engine.loss(state.params, test_examples)['loss']
                result = None
                conditions = None
                if not final_fit:
                    result, conditions = self.evaluate(state.params, directory, identity, epoch)
                    for label, values in conditions.items():
                        metrics.update({'validation/' + label + '/' + key: value for key, value in values.items()})
                checkpoint = self.save_checkpoint(state, identity, epoch, {'configuration': settings,
                    'epoch': epoch, 'step': step, 'validation': None if result is None else asdict(result),
                    'study_configuration': self.protocol, 'source_manifest_sha256': verify_source_manifest(SOURCE)})
                record = {'configuration': settings, 'checkpoint': checkpoint, 'metrics': metrics,
                          **({} if result is None else asdict(result))}
                trial_records.append(record)
                if result is not None:
                    self.records.append(record)
                    key = None if self.reference is None else balanced_selection_key(result, self.reference)
                    improved = result.zero_shot_nll < best_loss or (key is not None and (best_key is None or key < best_key))
                    if improved:
                        last_improvement = epoch
                    if result.zero_shot_nll < best_loss:
                        best_loss = result.zero_shot_nll
                    if key is not None and (best_key is None or key < best_key):
                        best_key = key
                    if self.minimum_loss is None or result.zero_shot_nll < self.minimum_loss['zero_shot_nll']:
                        self.minimum_loss = record
                    if key is not None and (self.selected is None or key < balanced_selection_key(
                            ValidationResult(**{field: self.selected[field] for field in ValidationResult.__dataclass_fields__}), self.reference)):
                        self.selected = record
                metrics['elapsed_epoch_seconds'] = time.monotonic() - started
                durations.append(metrics['elapsed_epoch_seconds'])
                self.epoch_durations.setdefault(settings['five_shot_training_fraction'], []).append(metrics['elapsed_epoch_seconds'])
                save_object(directory / f'epoch{epoch:02d}.json', record)
                emit(metrics)
                if not final_fit and identity != 'fixed-reference' and should_stop(epoch, last_improvement,
                        settings['minimum_epochs'], settings['patience']):
                    break
            summary = {'status': 'completed' if trial_records else 'interrupted', 'id': identity,
                'configuration': settings, 'completed_epochs': len(trial_records), 'epochs': trial_records,
                'test_used_for_selection': False}
            save_object(directory / 'summary.json', summary)
            events.write(json.dumps({'event': 'completed', 'summary': {'completed_epochs': len(trial_records)}}) + '\n')
            events.flush()
        del state, parallel_step
        gc.collect()
        return summary


def finalize_model(study: MatchedStudy, final: dict[str, Any]) -> None:
    """Export exact float32 weights and run the original full-test harness after selection is locked."""
    import jax
    import numpy as np
    import torch
    from flax import serialization
    from flax.traverse_util import flatten_dict
    from transformers import AutoModelForCausalLM
    from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model
    from train_gsm8k import file_record, mirror_directory, tied_parameters
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from benchmark_policy import verify_benchmark_samples
    require_search_time(study.budget, 60)
    selected_epoch = final['epochs'][-1]
    checkpoint = selected_epoch['checkpoint']
    state_path = Path(checkpoint['local_staging_directory']) / 'training_state.msgpack'
    if file_record(state_path) != checkpoint['manifest']['files'][0]:
        raise ValueError('Final optimizer checkpoint failed its SHA-256 verification.')
    restored = serialization.msgpack_restore(state_path.read_bytes())
    assert isinstance(restored, dict)
    expected_steps = math.ceil(7473 / final['configuration']['batch_size']) * final['completed_epochs']
    if int(restored['step']) != expected_steps:
        raise ValueError('Final checkpoint is not at the locked final optimizer update.')
    parameters = tied_parameters(restored['params'])
    del restored
    pytorch_model = AutoModelForCausalLM.from_config(study.engine.backend.config,
        torch_dtype=torch.float32, attn_implementation='eager')
    pytorch_model = load_flax_weights_in_pytorch_model(pytorch_model, parameters)
    pytorch_model.tie_weights()
    if any(parameter.dtype != torch.float32 for parameter in pytorch_model.parameters()):
        raise ValueError('Export reduced trained weight precision.')
    model_directory = study.output / 'balanced-final-model-fp32'
    flax_directory = study.output / 'balanced-final-flax'
    model_directory.mkdir()
    flax_directory.mkdir()
    pytorch_model.save_pretrained(model_directory, safe_serialization=True)
    study.engine.backend.tokenizer.save_pretrained(model_directory)
    reloaded = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float32, attn_implementation='eager')
    for name, value in pytorch_model.state_dict().items():
        if not torch.equal(value, reloaded.state_dict()[name]):
            raise ValueError('Saved PyTorch weights differ from the trained float32 model.')
    del reloaded
    study.engine.backend.config.save_pretrained(flax_directory)
    study.engine.backend.tokenizer.save_pretrained(flax_directory)
    (flax_directory / 'flax_model.msgpack').write_bytes(serialization.to_bytes(parameters))
    # Independent PyTorch forward verifies the exported tied model before the test benchmark.
    prompts = ['Question: What is 2 + 2?\nAnswer:',
        'Arithmetic example: 7 + 2 = 9.\n' * 100 + 'Question: What is 8 + 3?\nAnswer:']
    tokenizer = study.engine.backend.tokenizer
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='np')
    positions = inputs['attention_mask'].cumsum(axis=-1) - 1
    positions[inputs['attention_mask'] == 0] = 1
    with torch.inference_mode():
        expected = pytorch_model(torch.from_numpy(inputs['input_ids']),
            attention_mask=torch.from_numpy(inputs['attention_mask']),
            position_ids=torch.from_numpy(positions), use_cache=False).logits[:, -1, :].numpy()
    device_parameters = jax.device_put(parameters, study.engine.backend.devices[0])
    forward = jax.jit(lambda weights, tokens, mask, position_ids: study.engine.backend.model(tokens,
        attention_mask=mask, position_ids=position_ids, params=weights).logits[:, -1, :])
    observed = np.asarray(forward(device_parameters, inputs['input_ids'], inputs['attention_mask'], positions))
    difference = float(np.max(np.abs(expected - observed)))
    if difference >= .02 or not np.array_equal(expected.argmax(-1), observed.argmax(-1)):
        raise ValueError(f'Export numerical verification failed: {difference}')
    saved_parameters = serialization.msgpack_restore((flax_directory / 'flax_model.msgpack').read_bytes())
    for key, value in flatten_dict(parameters).items():
        np.testing.assert_array_equal(np.asarray(value), flatten_dict(saved_parameters)[key])
    save_object(study.output / 'export-verification.json', {'status': 'passed',
        'maximum_absolute_logit_difference': difference, 'logit_tolerance': .02,
        'flax_serialization_exact': True, 'pytorch_export_dtype': 'float32', 'step': expected_steps})
    for directory in (model_directory, flax_directory):
        save_object(directory / 'complete.json', {'status': 'complete', 'step': expected_steps,
            'source_checkpoint': checkpoint['gcs_directory'], 'source_checkpoint_sha256': checkpoint['manifest']['files'][0]['sha256'],
            'files': [file_record(path) for path in sorted(directory.iterdir()) if path.is_file()]})
        mirror_directory(directory, study.configuration['artifact_bucket_prefix'].removesuffix('/checkpoints'))
    del pytorch_model, saved_parameters, device_parameters, forward
    gc.collect()
    study.engine.backend.parameters = parameters
    study.engine.backend.replicated_parameters = jax.device_put_replicated(parameters, study.engine.backend.devices)
    study.engine.backend.pretrained = str(model_directory)
    study.engine.backend.revision = 'matched-validation-final'
    benchmark_results = {}
    for shots in (0, 5):
        require_search_time(study.budget, 60)
        study.engine.backend.progress_path = study.output / f'test-{shots}shot-generations.jsonl'
        results = simple_evaluate(model=study.engine.backend, tasks=['gsm8k_pinned'], num_fewshot=shots,
            task_manager=TaskManager(include_path=str(SOURCE / 'tasks')), batch_size=8, device='tpu',
            gen_kwargs='max_gen_toks=256,do_sample=False,temperature=0.0',
            random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
            log_samples=True)
        assert isinstance(results, dict)
        if results['n-samples']['gsm8k_pinned']['effective'] != 1319:
            raise ValueError('Benchmark did not cover the complete 1319-question test set.')
        samples = results.pop('samples')
        assert isinstance(samples, dict)
        verification = verify_benchmark_samples(samples['gsm8k_pinned'], results['results']['gsm8k_pinned'],
            read_object(SOURCE / 'baseline-hashes.json'), shots)
        save_object(study.output / f'test-{shots}shot-verification.json', verification)
        with (study.output / f'test-{shots}shot-samples.jsonl').open('x') as handle:
            for sample in samples['gsm8k_pinned']:
                handle.write(json.dumps(sample, default=str) + '\n')
        save_object(study.output / f'test-{shots}shot-results.json', json.loads(json.dumps(results, default=str)))
        benchmark_results[str(shots)] = results['results']['gsm8k_pinned']
    save_object(study.output / 'study-completed.json', {'status': 'completed', 'test_used_for_selection': False,
        'final_model_directory': str(model_directory), 'benchmarks': benchmark_results,
        'final_train_loss': selected_epoch['metrics']['train/loss'], 'final_test_loss': selected_epoch['metrics']['test/loss'],
        'finished_unix_seconds': time.time(), 'remaining_seconds': study.budget.remaining(time.time())})


def main() -> None:
    """Rebuild the fixed reference, search the predeclared grid, and lock the refit recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id', required=True)
    arguments = parser.parse_args()
    if not arguments.attempt_id or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for character in arguments.attempt_id):
        raise ValueError('Invalid attempt identity.')
    protocol = read_object(ROOT / 'protocol.json')
    receipt = read_object(ROOT / 'execution-receipt.json')
    budget = verify_execution_receipt(receipt, protocol, ROOT / 'source', socket.gethostname(), time.time())
    verify_source_manifest(ROOT / 'source')
    verify_source_manifest(SOURCE)
    if ROOT != Path('/home/kristinelu/smollm2-gsm8k/matched-validation-20261103'):
        raise PermissionError('Run only from the authorized VM-local snapshot.')
    output = ROOT / 'output' / ('search-' + arguments.attempt_id)
    output.mkdir(exist_ok=False)
    study = MatchedStudy(protocol, budget, output)
    predecessor = ROOT / 'output/search-first'
    predecessor_terminal = read_object(ROOT / 'control/search-first/finished.json')
    if predecessor_terminal['status'] != 'failed' or predecessor_terminal['exit_code'] != 1:
        raise ValueError('This recovery is bound to the recorded zero-step HBM failure.')
    reference_summary = read_object(predecessor / 'fixed-reference/summary.json')
    control_summary = read_object(predecessor / 'matched-01/summary.json')
    if reference_summary['completed_epochs'] != 3 or control_summary['completed_epochs'] != 7:
        raise ValueError('Completed predecessor work differs from the documented recovery.')
    record = read_object(predecessor / 'frozen-reference.json')
    study.records = reference_summary['epochs'] + control_summary['epochs']
    study.minimum_loss = min(study.records, key=lambda item: item['zero_shot_nll'])
    study.epoch_durations[0.0] = [item['metrics']['elapsed_epoch_seconds'] for item in study.records]
    save_object(output / 'recovered-predecessor.json', {'path': str(predecessor),
        'status': 'failed-before-first-matched-02-update', 'reused_completed_trials': ['fixed-reference', 'matched-01'],
        'effective_batch_size': 32, 'mixed_prompt_microbatch_size': 8, 'mixed_prompt_accumulation_steps': 4,
        'original_budget_start': budget.started_unix_seconds, 'deadline_unix_seconds': budget.started_unix_seconds + budget.authorized_seconds,
        'reason': 'Native full-batch long-context backward required 21.76 GiB HBM per chip; accumulate four microbatches before one optimizer update.'})
    if abs(record['zero_shot_nll'] - protocol['reference']['expected_original_validation_nll']) > 1e-4:
        raise ValueError('The recreated epoch-three loss reference did not reproduce the original NLL.')
    study.reference = ValidationResult(**{field: record[field] for field in ValidationResult.__dataclass_fields__})
    study.reference_record = record
    study.selected = record
    for previous in study.records:
        previous_result = ValidationResult(**{field: previous[field] for field in ValidationResult.__dataclass_fields__})
        key = balanced_selection_key(previous_result, study.reference)
        selected_result = ValidationResult(**{field: study.selected[field] for field in ValidationResult.__dataclass_fields__})
        if key is not None and key < balanced_selection_key(selected_result, study.reference):
            study.selected = previous
    save_object(output / 'frozen-reference.json', record)
    audit_completion = read_object(ROOT / 'output/audit-first/completed.json')
    for audited in audit_completion['results']:
        if audited['protocol_sha256'] != study.digest:
            raise ValueError('Historical checkpoint audit used different validation prompts.')
        old_summary = read_object(Path('/home/kristinelu/smollm2-gsm8k/research-20261102/output') /
            audited['trial_id'] / 'summary.json')
        validation = ValidationResult('audit-' + audited['trial_id'], audited['epoch'], 747,
            audited['zero_shot']['strict_correct'], audited['five_shot']['strict_correct'],
            audited['zero_shot']['nll'], audited['five_shot']['nll'], study.digest)
        candidate = {**asdict(validation), 'configuration': {**protocol['candidate_defaults'],
            **old_summary['configuration'], 'id': validation.trial_id, 'five_shot_training_fraction': 0.0},
            'checkpoint': {'original_local_directory': audited['checkpoint'], 'manifest': audited['checkpoint_manifest']}}
        study.records.append(candidate)
        key = balanced_selection_key(validation, study.reference)
        selected_validation = ValidationResult(**{field: study.selected[field] for field in ValidationResult.__dataclass_fields__})
        if key is not None and key < balanced_selection_key(selected_validation, study.reference):
            study.selected = candidate
        if validation.zero_shot_nll < study.minimum_loss['zero_shot_nll']:
            study.minimum_loss = candidate
    for declared in protocol['candidates']:
        if declared['id'] == 'matched-01':
            continue
        specification = {**declared, 'canonical_candidate_id': declared['id']}
        if specification['id'] == 'matched-02':
            specification['id'] = 'matched-02-r1'
        fraction = specification['five_shot_training_fraction']
        measured = study.epoch_durations.get(fraction, [])
        estimate = 300.0 + protocol['candidate_defaults']['minimum_epochs'] * epoch_admission_seconds(measured)
        save_object(output / (specification['id'] + '-admission.json'), {'remaining_seconds': budget.remaining(time.time()),
            'estimated_minimum_trial_seconds': estimate, 'measured_epoch_seconds': measured,
            'finalization_reserve_seconds': study.reserve, 'admitted': budget.admits(time.time(), estimate, study.reserve)})
        if not budget.admits(time.time(), estimate, study.reserve):
            break
        if shutil.disk_usage(output).free < protocol['budget']['minimum_free_disk_gib'] * 1024**3:
            raise OSError('Node root disk is below the fixed reserve.')
        study.train_trial(specification, protocol['candidate_defaults']['maximum_epochs'])
    selected = study.selected
    assert selected is not None
    lock = {'status': 'locked', 'selected': selected, 'minimum_zero_shot_nll': study.minimum_loss,
        'reference': study.reference_record, 'frontier': nondominated(study.records), 'test_used_for_selection': False,
        'selection_locked_unix_seconds': time.time(), 'remaining_seconds': budget.remaining(time.time()),
        'refit_recipe': {**selected['configuration'], 'id': 'balanced-final', 'epochs': selected['epoch']}}
    save_object(output / 'locked-selection.json', lock)
    print(json.dumps({'event': 'selection_locked', 'trial_id': selected['trial_id'], 'epoch': selected['epoch'],
        'zero_shot_correct': selected['zero_shot_correct'], 'five_shot_correct': selected['five_shot_correct'],
        'zero_shot_nll': selected['zero_shot_nll'], 'five_shot_nll': selected['five_shot_nll']}), flush=True)
    study.reserve = 900
    final = study.train_trial(lock['refit_recipe'], selected['epoch'], final_fit=True)
    if final['completed_epochs'] != selected['epoch']:
        raise TimeoutError('Final refit did not complete its locked epoch count.')
    save_object(output / 'refit-completed.json', {'status': 'completed', 'selection': lock, 'final': final,
        'finished_unix_seconds': time.time()})
    finalize_model(study, final)


if __name__ == '__main__':
    main()

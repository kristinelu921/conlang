# Conlang JAX v5e-8 batch-256 full run 2 retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| Status | `completed`; throughput gate failed |
| Started | `2026-10-15T00:36:17Z` |
| Finished | `2026-10-15T00:41:06Z` |
| Baseline | `20261014-conlang-jax-v5e8-b256-full1`; retry of `20261015-conlang-jax-v5e8-b256-full2` |
| Git revision | `e94f185` on `codex/jax-tpu-training` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a --project=he-vision-group --zone=us-west4-a --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-full2-retry1-20261015T003437.703532461Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-full2-retry1.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2-retry1.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, `2x4`, 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-full2-retry1.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-full2-retry1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 1000
eval_every: 100
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2-retry1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-full2-retry1
wandb_enabled: false
hidden_size: 128
message_length: 4
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 2
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Failed attempt | Current | Source |
|---|---:|---:|---|
| `wandb_enabled` | true | false | failed-attempt and retry YAMLs |

Run identity, output path, and W&B display name are fresh. All scientific settings are identical to the failed seed-1730 attempt; W&B is disabled solely to bypass its deterministic oversized-histogram failure while retaining node-local metrics and outputs.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | baseline median steady-state throughput | 8079.944432 examples/s | 99 post-warmup intervals through step 1000 | higher is better | `20261014-conlang-jax-v5e8-b256-full1` |
| training | sustained-throughput floor | 6463.955546 examples/s | post-warmup retry intervals | higher is better | 20% regression tolerance from the baseline full run |
| training | median steady-state throughput | 5468.273897 examples/s | 99 post-warmup intervals through step 1000 | higher is better | archived `metrics.jsonl` |
| training | mean steady-state throughput | 5447.970944 examples/s | 99 post-warmup intervals through step 1000 | higher is better | archived `metrics.jsonl` |
| training | throughput range | 4264.559084–5553.937660 examples/s | 99 post-warmup intervals through step 1000 | higher is better | archived `metrics.jsonl` |
| training | intervals meeting throughput floor | 0 / 99 | steps 20–1000 | higher is better | archived `metrics.jsonl` |
| training | median relative to baseline | -32.323% | steps 20–1000 | higher is better | derived from archived metrics and baseline |
| training | median relative to allowed floor | -15.404% | steps 20–1000 | higher is better | derived from archived metrics and floor |
| training | loss | 4.103382111 | step 1000 | lower is better | archived `metrics.jsonl` |
| validation | exact match | 0.042968750 | step 1000 | higher is better | archived `metrics.jsonl` |
| validation | mean joint log probability | -4.065857887 | step 1000 | higher is better | archived `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-full2-retry1.yaml`; archived effective copy SHA-256 `dada29a6c65d3be4716b915026ae8be223e61b53272d94b5b0dcc43a3fb53b85` | yes; parsed, matched scientifically to the failed attempt, and committed alone as `e94f185` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; retry config commit `e94f185` pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-full2-retry1-20261015T003437.703532461Z`; remote `/home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-full2-retry1-20261015T003437.703532461Z` | yes; 31 files, 310848 bytes, aggregate SHA-256 `2ce9e9604cca012ab2dfc3e99737d36f5207afa0219a967eb94b43a9998bd10f` matched locally and remotely |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` on the TPU, sourced from the `us-west4` Kristine bucket | yes; 7 files, 11643131 bytes, 125000 verified questions; training used 100000 examples and validation used 256 |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2-retry1/checkpoint.msgpack` | yes; 22656218 bytes, SHA-256 `24d6996f0efd62ea23a2efe2b54045cb8adbc6c69ed5652a747efb47d5fe1ac6`, restored payload step `1000` |
| Tracking run | disabled for this telemetry-recovery retry | not applicable |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-full2-retry1/metrics.jsonl` | yes; 2780328 bytes, 110 rows, SHA-256 `b820088e509f15136f32d31a8b7b6b3df4e0fe757df3400cac37bb677d9cce5b` |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-full2-retry1/outputs.jsonl` | yes; 14017135 bytes, SHA-256 `7263b6dc580bd7a366888169e7c96a69834ad90e9b08479c5f869ce7fcd5a7be` |
| Canonical log | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-full2-retry1/20261015-conlang-jax-v5e8-b256-full2-retry1.launcher.log.gz` | yes; deterministic gzip SHA-256 `38c7512df9ae86f2f39e762c4cd51d99708654671951ae8eff2a3d069d6a5b50`, decompressed SHA-256 `a93c3f3e111f70613abac5b5bd564919e2a35c674275dec9e63c3e5c658123b6` matched the remote log |

## Failure Details

- Last known step: `1000`; `TRAINING_COMPLETE` emitted
- Error summary: no runtime error; the run completed, but its sustained-throughput gate failed
- Detailed output: canonical log and node-local metrics are archived above

## Observations

- The failed attempt completed step 10, then W&B rejected a vocabulary-sized histogram before step 20. The model, optimizer, regional data, TPU backend, and compile all passed their gates.
- This retry changes only fresh identities and disables W&B. Full metrics, per-step model outputs, the foreground log, and terminal checkpoint remain node-local artifacts for direct monitoring and archival.
- The retry emitted 100 training metric rows and 10 validation rows, completed all 1000 steps, and wrote a restorable step-1000 checkpoint.
- Median post-warmup throughput was 5468.273897 examples/s. All 99 measured intervals were below the 6463.955546 examples/s floor, so functional completion does not satisfy the performance gate.
- The terminal all-worker audit found the exact trainer absent, no VFIO holder, and no run lock. The TPU remained `READY` and `HEALTHY` after completion.

## Interpretation

- This is an infrastructure-only retry of the seed-1730 replica. Disabling W&B avoids the confirmed telemetry failure without altering optimization, model shape, data, precision, RNG seed, or evaluation schedule.
- The run is valid for result replication but not for accepting the current instrumentation path as throughput-neutral. The immutable source snapshot includes the preserved local telemetry worktree, so its additional histogram work is a plausible contributor to the regression; that causal attribution remains an inference rather than a measured ablation.

## Updates

- 2026-10-15T00:30:09Z: Planned a fresh retry identity after preserving the failed step-10 attempt. The retry disables W&B and retains all scientific settings; config validation, mandatory capacity selection, commit, PR update, immutable staging, and launch remain pending.
- 2026-10-15T00:33:31Z: The complete retry YAML parsed and differed from the failed attempt only in fresh identities and `wandb_enabled: false`; it was committed alone as `e94f185`. The fresh mandatory inventory found the prior v5e-8 occupied by an unrelated, healthy UID-2001 InstaManip smoke. Its owner, PID `2160637`, elapsed time, full command, and run identity were recorded read-only and the workload was left untouched. The remaining available Kristine v4-8s lack the pinned runtime and Central2 regional dataset, so the orchestrator is waiting for the proven v5e path rather than silently changing hardware and data provenance.
- 2026-10-15T00:34:37Z: A fresh mandatory inventory returned the exact v5e-8 in its final `AVAILABLE` table. The source was snapshotted, hashed, and copied to node-local storage; runtime, regional data, effective config, output freshness, API health, and VFIO idleness passed before launch.
- 2026-10-15T00:36:17Z: Launched the immutable snapshot in the foreground on the 8-device JAX TPU backend.
- 2026-10-15T00:41:06Z: The run emitted `TRAINING_COMPLETE` at step 1000 and wrote the final checkpoint. The terminal all-worker audit verified TPU health, trainer exit, idle VFIO, checkpoint payload step, and artifact hashes. The completed run failed the sustained-throughput gate and requires a separate optimization/measurement pass before performance acceptance.

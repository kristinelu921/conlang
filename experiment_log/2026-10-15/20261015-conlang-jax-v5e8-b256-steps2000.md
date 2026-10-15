# Conlang JAX v5e-8 batch-256 2,000-step run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-steps2000` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| Git revision | reviewed config commit `2f299c5298f9cc7652b28a060446ac0b14128837`; local experiment merge `23f8e5d95a87e0344848cf02a29050f0c1cdb8f1` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a --project=he-vision-group --zone=us-west4-a --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-steps2000.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, topology `2x4`, 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-steps2000.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-steps2000
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 100
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-steps2000
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| `steps` | 1000 | 2000 | baseline and selected run YAMLs |

Run ID, output path, and W&B display name are independently fresh. All model, optimizer, batch, precision, data, evaluation, logging, telemetry, and RNG settings remain matched to the successful seed-1730 retry.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | immediate-baseline median steady-state throughput | 5468.273897 examples/s | 99 post-warmup intervals through step 1000 | higher is better | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| training | like-for-like sustained-throughput floor | 4374.619118 examples/s | post-warmup intervals | higher is better | 20% regression tolerance from the immediate baseline |
| training | prior full-run acceptance floor | 6463.955546 examples/s | post-warmup intervals | higher is better | retained context from `20261014-conlang-jax-v5e8-b256-full1` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-steps2000.yaml` | yes; parsed and validated, SHA-256 `c3d1f6563c27c6e649be739441bb608a11e4709b77f98f9beb252a740c166409`, committed alone as `2f299c5` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed config commit `2f299c5` pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z`; remote `/home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z` | yes; 32 files, 311663 bytes, sorted SHA-256-manifest digest `bca20aa8e3383acf7cbaeb3fc7ea6ab1326512450fee1a5cdd6e18e864e80f3a` matched locally and remotely |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, sourced from `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; bucket location `US-WEST4`; 7 files, 11643131 bytes, 15625 worlds and 125000 questions passed complete manifest verification |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/checkpoint.msgpack` | no; run not started |
| Tracking run | disabled because the prior W&B histogram path deterministically rejected vocabulary-sized telemetry | not applicable |
| Metrics | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/metrics.jsonl` | no; run not started |
| Evaluation outputs | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/outputs.jsonl` | no; run not started |
| Canonical log | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000.launcher.log` | no; run not started |

## Failure Details

- Last known step: not applicable; run not started
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The complete CPU suite passed under the project interpreter: 88 tests and 7 subtests.
- The correctness and cleanup reviews reported no confirmed findings for base `e94f185` through reviewed head `2f299c5`.
- The selected TPU is currently `READY`, `HEALTHY`, Spot, runtime-ready, and `AVAILABLE`; preflight found 4 GiB free on both relevant filesystems, idle VFIO, no active trainer, a fresh output identity, and writable `/tmp/tpu_logs`.
- The immutable snapshot contains the same `train.py` bytes as the successful 1,000-step retry: SHA-256 `c6c5cd7351a017f60c9a584bc4d989b8c165a6367a7f6c499bca0b07919f6dac`.

## Interpretation

- This run tests the effect of doubling only the training horizon under the successful retry's exact scientific and telemetry settings.
- The immediate baseline is the like-for-like throughput comparator. The older full-run floor remains useful context: completing this run does not by itself resolve the already documented telemetry-era performance regression.

## Updates

- 2026-10-15T01:40:00Z: Planned and validated the 2,000-step run. The fresh mandatory inventory selected the runtime-ready us-west4 Kristine v5e-8 from the final `AVAILABLE` table. The reviewed one-file config commit was pushed to PR 1, the immutable source was copied once and matched locally/remotely, and regional dataset integrity passed before launch.

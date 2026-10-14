# Conlang JAX v5e-8 b128 gate 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b128-gate1` |
| Status | `staged` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | not specified |
| Git revision | `cbefd6190fab0f9969c55c751e36e678c5bf39ee` on `codex/jax-tpu-training` |
| Command | `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 /home/kristinelu/elf-runs/runtime/venv/bin/python train.py --config configs/conlang-jax-v5e8-gate1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8; single host; READY/HEALTHY and AVAILABLE at 2026-10-14 preflight |

## Configuration

Source: `configs/conlang-jax-v5e8-gate1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b128-gate1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 128
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 200
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b128-gate1
wandb_enabled: true
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

No confirmed changes; no explicit experimental baseline was specified. The b128 configuration is a bounded throughput gate rather than a scientific comparison.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| not recorded | not recorded | not recorded | not recorded | not recorded | run has not launched |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-gate1.yaml` | yes |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261014-conlang-jax-v5e8-b128-gate1-20261014T211431.116766090Z`; aggregate SHA-256 `45a0e2fa460c726db92fc77ee10031b9e48cfa6428f50ea1b7a685f91c21deb6` | yes |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` localized to `/home/kristinelu/elf-data/conlangs-v1`; 125,000 questions | yes |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/checkpoint.msgpack` | no |
| Tracking run | not recorded | no |
| Metrics | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/metrics.jsonl` | no |

## Failure Details

- Last known step: not applicable
- Error summary: one zero-step data-staging verification attempt used the nonexistent stale interpreter path `/home/kristinelu/elf-venv/bin/python`
- Detailed output: the regional copy completed into a temporary directory; verification was retried with `/home/kristinelu/elf-runs/runtime/venv/bin/python`, passed the complete manifest and checksum audit, and the directory was promoted atomically. No trainer, W&B run, output directory, or TPU client started.

## Observations

- The selected TPU was READY, HEALTHY, runtime-ready, and idle in the repository-required filtered inventory.
- `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` is in `US-WEST4` and contained the manifest plus all six compressed splits, totaling 11,643,131 bytes during preflight.
- CPU tests passed for static token shapes, pmapped parameter updates, bfloat16 compute with a float32 loss, relative-position attention, YAML field validation, and TPU path validation.

## Interpretation

- Static shapes, one-time host tokenization, and all-local-device data parallelism are expected to remove the major avoidable JAX recompilation and Python collation costs. Actual TPU throughput remains unmeasured until the immutable source can be pushed and reviewed.

## Updates

- 2026-10-14T20:51:22Z: Created the planned report after selecting the same-region v5e-8 and preparing the native JAX throughput gate; launch remains blocked before staging.
- 2026-10-14T21:10:29Z: Pushed reviewed code/config commit `cbefd6190fab0f9969c55c751e36e678c5bf39ee` and opened PR 1; the run remains unlaunched pending fresh preflight and immutable staging.
- 2026-10-14T21:16:55Z: Preflight passed with 4 GiB free and a 1 GiB minimum. Copied the 27-file immutable source snapshot to node-local storage with matching size and aggregate SHA-256. Localized the complete US-WEST4 dataset and verified all 125,000 questions after correcting the zero-step verification interpreter path. The trainer remains unlaunched.

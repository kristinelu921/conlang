# Conlang JAX v5e-8 b256 optimization 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b256-opt1` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261014-conlang-jax-v5e8-b128-gate1` |
| Git revision | `3cd53e9c0d656dc9bc5634511dff405017960515` on `codex/jax-tpu-training` |
| Command | `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 /home/kristinelu/elf-runs/runtime/venv/bin/python train.py --config configs/conlang-jax-v5e8-b256-opt1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8; single host |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-opt1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b256-opt1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
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
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b256-opt1
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

- `batch_size`: 128 -> 256. Every other model, optimizer, precision, data, seed, sequence-length, evaluation, and hardware setting is unchanged.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| not recorded | not recorded | not recorded | not recorded | not recorded | run has not launched |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-opt1.yaml` | yes |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/checkpoint.msgpack` | no |
| Tracking run | not recorded | no |
| Metrics | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/metrics.jsonl` | no |

## Failure Details

- Last known step: not applicable
- Error summary: none; run has not launched
- Detailed output: not recorded

## Observations

- The batch-128 baseline completed at a median 5,193.3586 examples/s across 19 post-warmup intervals, ranging from 5,160.9233 to 5,208.0937 examples/s.
- The same node-local, fully verified US-WEST4 dataset will be reused without another regional transfer.

## Interpretation

- This arm tests whether doubling the per-device batch from 16 to 32 improves TPU utilization. It is promoted only if steady-state examples/s improves without non-finite loss, memory failure, or a material training-quality regression.

## Updates

- 2026-10-14T21:23:45Z: Selected the directly comparable batch-256 optimization arm after the batch-128 run completed and passed its terminal audit.
- 2026-10-14T21:25:36Z: Pushed the selected run config and report in commit `3cd53e9c0d656dc9bc5634511dff405017960515` to PR 1; the optimization arm remains unlaunched pending fresh inventory, preflight, and immutable staging.

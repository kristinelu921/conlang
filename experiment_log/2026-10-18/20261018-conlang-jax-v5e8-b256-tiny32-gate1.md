# Conlang JAX v5e-8 batch-256 tiny-32 gate 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-gate1` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | not specified; operational throughput reference only: `20261014-conlang-jax-v5e8-b256-opt1` |
| Git revision | pending configuration commit |
| Pull request | pending |
| Command | not recorded; will be pinned after immutable staging and before launch |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; `us-west4-a`; `v5litepod-8`; single host; Spot name; API `READY`/`HEALTHY`; occupancy `AVAILABLE` |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-gate1.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-gate1
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
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-gate1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-gate1
wandb_enabled: true
hidden_size: 32
message_length: 1
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

No explicit experimental baseline was specified. This is a throughput and correctness gate for the newly requested single-vector, 32-dimensional, two-layer architecture and single-link dataset schema, not a controlled scientific comparison.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | steady-state throughput | pending | post-warmup | max | pending |
| validation | exact match | pending | 200 | max | pending |
| validation | mean joint log probability | pending | 200 | max | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-gate1.yaml` | yes; parsed locally, commit pending |
| Source snapshot | pending | no |
| Effective configuration | pending | no |
| Checkpoint | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-gate1/checkpoint.msgpack` | no; run not launched |
| Metrics | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-gate1/metrics.jsonl` | no; run not launched |
| Model outputs | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-gate1/outputs.jsonl` | no; run not launched |
| Canonical log | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-gate1.launcher.log` | no; run not launched |
| Tracking run | not recorded | no |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The mandatory `kristine`-filtered inventory completed successfully and listed the selected v5e-8 in its final `AVAILABLE` table with runtime `READY`.
- Exact API inspection reported `v5litepod-8`, `READY`, and `HEALTHY`; the read-only all-worker occupancy helper reported `AVAILABLE`.
- The instantiated model has 228,271 parameters, one 32-dimensional continuous vector per sender, two sender layers, two receiver-encoder layers, and two receiver-decoder layers.
- Aggregate parameter, optimizer, and gradient storage is far below one GiB; activation storage at global batch 256 is expected to fit the selected eight-chip topology with substantial headroom. Remote preflight remains authoritative.
- The previous batch-256 v5e-8 gate recorded 7,900.3184 examples/s, but it used a different model and dataset schema and is only an operational regression reference.

## Interpretation

- Passing the first completed step proves launch viability; promotion requires several steady-state post-warmup intervals, finite metrics, successful final evaluation/checkpointing, and no material unexplained throughput regression.
- A successful gate will be followed by a distinct full-run identity and configuration rather than reusing this output directory.

## Updates

- 2026-10-18T21:58:25-04:00: User authorized launch and training. Created this planned report after the required filtered inventory, exact API inspection, and occupancy check selected the runtime-ready `kristine` v5e-8.

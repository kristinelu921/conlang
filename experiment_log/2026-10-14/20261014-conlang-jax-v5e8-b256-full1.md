# Conlang JAX v5e-8 batch-256 full run 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b256-full1` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261014-conlang-jax-v5e8-b256-opt1` |
| Git revision | `b337b68be31f36a314e0dae5720a72813c598ddd` on `codex/jax-tpu-training` |
| Command | `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 /home/kristinelu/elf-runs/runtime/venv/bin/python train.py --config configs/conlang-jax-v5e8-b256-full1.yaml` |
| Seed | `1729` |
| Hardware | selected candidate: `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8; single host; launch readiness pending final preflight |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-full1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b256-full1
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
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b256-full1
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| `steps` | 200 | 1000 | baseline and full-run YAMLs |
| `eval_every` | 200 | 100 | baseline and full-run YAMLs |

Run identity, output path, and W&B display name are fresh. Model shape, optimizer, batch size, precision, dataset, seed, and topology remain identical to the accepted optimization gate.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | accepted promotion-gate median throughput | 7900.318449 examples/s | 19 post-warmup intervals through step 200 | higher is better | `20261014-conlang-jax-v5e8-b256-opt1` |
| training | sustained-throughput floor | 6320.254759 examples/s | post-warmup full-run intervals | higher is better | 20% regression tolerance from the accepted gate |
| validation | exact match | not recorded | not started | higher is better | pending |
| validation | mean joint log probability | not recorded | not started | higher is better | pending |
| training | final loss | not recorded | not started | lower is better | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-full1.yaml` | yes; parsed and validated locally |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing; update pending |
| Source snapshot | pending immutable staging under `/kmh-nfs-ssd-us-mount/staging/kristine/elf/` | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`; source `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | previously verified; current preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1/checkpoint.msgpack` | no; not started |
| Tracking run | not recorded | no; not started |
| Metrics | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1/metrics.jsonl` | no; not started |
| Canonical log | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1.launcher.log` | no; not started |

## Failure Details

- Last known step: not applicable; not started
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The mandatory filtered inventory selected the available, runtime-ready US-WEST4 v5e-8 that completed the accepted batch-256 gate.
- The accepted gate used the same explicit `message_length: 4`, model dimensions, precision, batch size, seed, regional data, and TPU topology planned here.

## Interpretation

- This is the promoted full run from the accepted batch-256 throughput gate. A sustained post-warmup throughput below 80% of the matched 7900.318449 examples/s median is a material regression requiring diagnosis rather than silent continuation.

## Updates

- 2026-10-14T21:41:23Z: User authorized launch. The mandatory filtered inventory returned the selected v5e-8 as `AVAILABLE` with ELF runtime `READY`. The complete YAML parsed successfully and was committed alone as `b337b68be31f36a314e0dae5720a72813c598ddd`; immutable staging, final preflight, and launch remain pending.

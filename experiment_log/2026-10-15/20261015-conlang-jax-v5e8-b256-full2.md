# Conlang JAX v5e-8 batch-256 full run 2

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-full2` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261014-conlang-jax-v5e8-b256-full1` |
| Git revision | `369e356c0dff0678c92f7aaa5dd944f7c0cf8a68` on `codex/jax-tpu-training`; run config introduced by `b493a2b5e5ce90c5259951e75c5423b11fad7ae3` |
| Command | `cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-full2-20261015T001755.627224388Z && JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=$PWD HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-full2.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2.launcher.log` |
| Seed | `1730` |
| Hardware | planned: `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8, topology 2x4; single host; Spot; filtered inventory reported `AVAILABLE` with ELF runtime `READY` |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-full2.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-full2
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-full2
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
| `seed` | 1729 | 1730 | baseline and replica YAMLs |

Run identity, output path, and W&B display name are fresh. All other scientific settings are identical to the baseline full run.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | baseline median steady-state throughput | 8079.944432 examples/s | 99 post-warmup intervals through step 1000 | higher is better | `20261014-conlang-jax-v5e8-b256-full1` |
| training | sustained-throughput floor | 6463.955546 examples/s | post-warmup replica intervals | higher is better | 20% regression tolerance from the baseline full run |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-full2.yaml` | yes; parsed, validated, and committed alone as `b493a2b5e5ce90c5259951e75c5423b11fad7ae3` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed head `369e356c0dff0678c92f7aaa5dd944f7c0cf8a68` pushed |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-full2-20261015T001755.627224388Z`; node-local copy under `/home/kristinelu/elf-runs/source/`; aggregate SHA-256 `befd4636c69084e2df64dbecd011830cd032a465052d801e6ade13ee451fd6ab` | yes; 30 files, 310,024 file bytes, exact local/remote match |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`; source `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; US-WEST4, 7 canonical objects, 11,643,131 file bytes |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2/checkpoint.msgpack` | no; run not started |
| Tracking run | not recorded | no |
| Metrics | not recorded | no |
| Canonical log | not recorded | no |

## Failure Details

- Last known step: not applicable; run not started
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The mandatory filtered inventory returned the selected US-WEST4 v5e-8 as `AVAILABLE` with ELF runtime `READY`.
- This replica explicitly pins `message_length: 4`, so unrelated uncommitted edits changing local defaults cannot change the selected run configuration.
- The staged snapshot captures those pre-existing uncommitted default/documentation edits, while the immutable replica YAML remains explicit and has SHA-256 `b7129126d22a8818e7640d7a698c8b4dc7a0c5af3ce26485f8e6e4333b94c9c9` locally and remotely.
- One diagnostic process probe falsely matched its own remote shell because the inline command contained both `train.py` and the run ID. The canonical occupancy helper remained `AVAILABLE`; a corrected executable-and-argument check found no trainer. This was a read-only diagnostic failure, not a launch or staging attempt.

## Interpretation

- This is an independent seed replica of the completed full run. A sustained post-warmup throughput below 80% of the matched 8079.944432 examples/s median is a material regression requiring diagnosis rather than silent continuation.

## Updates

- 2026-10-15T00:12:13Z: User requested one more run. The mandatory filtered inventory selected the available, runtime-ready US-WEST4 v5e-8. Planned a seed-1730 replica with all other scientific settings identical to full run 1.
- 2026-10-15T00:13:18Z: The complete TPU YAML parsed successfully and was committed alone as `b493a2b5e5ce90c5259951e75c5423b11fad7ae3`; immutable staging, final preflight, and launch remain pending.
- 2026-10-15T00:20:32Z: The correctness and cleanup reviews reported no confirmed findings at head `369e356c0dff0678c92f7aaa5dd944f7c0cf8a68`; the CPU suite passed (`91` tests and `7` subtests), and the reviewed head was pushed to PR 1. The immutable snapshot was copied exactly once to node-local storage with matching aggregate hash, count, bytes, entrypoint hash, and config hash. Fresh preflight verified `READY`/`HEALTHY`, Spot, idle VFIO and lock state, JAX 0.6.2 / Flax 0.10.5 / Optax 0.2.8, the verified regional dataset, and absent output/log paths.

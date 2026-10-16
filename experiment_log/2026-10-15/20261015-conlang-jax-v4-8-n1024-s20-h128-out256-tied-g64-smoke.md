# Conlang JAX v4-8 1,024-example hidden-128 tied-encoder smoke test

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64` |
| Git revision | reviewed source and configuration `ef22ee5a3c5d7cf45615315d5538dd7570eb510a` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke-20261016T035317.800843722Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, expected four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 20
eval_every: 10
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/checkpoint.msgpack
limit_examples: 1024
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: true
role_gate_channels: 64
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
| Training examples | 100000 | 1024 | selected config and dataset loader |
| Training steps | 2000 | 20 | selected config |
| Evaluation interval | 100 steps | 10 steps | selected config |
| Transformer/message hidden width | 512 | 128 | selected config |

The batch size 256, output hidden dimension 256, tied sender/receiver encoder, separate 64-channel role gates, optimization, seed, validation size, regional dataset, and compact vocabulary histograms remain unchanged. The 1,024-example limit is exactly four full batches per epoch.

## Results

Pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml` | yes; SHA-256 `e63b3b8086612cf0997ff20fdf6132600fa54ee2c21b62e06793ff045250976c` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed configuration pushed at `ef22ee5` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke-20261016T035317.800843722Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 42 files and 356842 bytes with relative SHA-256-manifest digest `58c1b291c2cc2a8cffe9c66465265d76a0de286666afd79f8f0b881091cc304c` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; deterministic Central2 dataset reverified with 15625 worlds and 125000 questions; manifest SHA-256 from the established record `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Runtime | `/home/kristinelu/elf-runs/runtime/venv` | yes; pinned JAX 0.6.2 profile passed a no-initialize readiness check |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/checkpoint.msgpack` | no; launch pending |
| Tracking run | not recorded | no; launch pending |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The mandatory Kristine-only inventory selected this v4-8 from the final `AVAILABLE` table. Direct checks reported `READY`, `HEALTHY`, and `AVAILABLE`.
- Remote preflight reported 86 GiB free for data and outputs, an absent run output path, and an idle TPU device.
- Correctness review found no issue in the exact one-file configuration commit. Cleanup's only suggestion was rejected because `20261015` is the current date in the required America/New_York client timezone despite the UTC rollover; no material review finding remains.

## Interpretation

- This smoke test is intended to validate the complete tied-encoder width-128 pipeline on a small deterministic prefix, not to estimate model quality or compare against the 2,000-step baseline.
- The paired untied-encoder arm must use the same data limit, seed, hardware, and all other hyperparameters before any shared-vs-untied interpretation.

## Updates

- `2026-10-16T03:54:22Z`: Reviewed config commit was pushed to PR 1, the existing runtime and regional dataset were reverified, the immutable source snapshot matched locally and remotely, and remote preflight passed. Launch remains pending the final occupancy gate.

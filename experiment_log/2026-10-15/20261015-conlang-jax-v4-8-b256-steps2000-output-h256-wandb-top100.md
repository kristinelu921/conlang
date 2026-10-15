# Conlang JAX v4-8 batch-256 2,000-step output-head-256 W&B top-100 vocabulary histogram

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100` |
| Status | `planned` |
| Started | pending |
| Finished | pending |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100` |
| Git revision | reviewed source and configuration `9fb9ce12a4e10bff4517a512b81b94143f40a36d` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=0 --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100-20261015T053225.239402639Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
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

| Parameter | Baseline | This run |
|---|---:|---:|
| Transformer/message `hidden_size` | 128 | 128 (unchanged) |
| Output head | direct 128-to-vocabulary projection | 128-to-256 ReLU hidden projection, then vocabulary projection |

All other scientific hyperparameters, the seed, W&B settings, and top-100 generated-vocabulary histogram telemetry match the baseline.

## Results

Pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml` | yes; SHA-256 `2ad1f7044133bdf0bcdd56d87b2bb1da20ee9c199daa47aa7139f1441a4a2503` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `9fb9ce1` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100-20261015T053225.239402639Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | local verified; 38 files, 332243 bytes, relative SHA-256-manifest digest `52a11a0e125da5550f1ffc5385d75ec8eaa2b2db27cb79544c9bb61935ca99e9`; remote verification pending |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | present in `us-central2`; full manifest verification pending |
| Tracking run | pending | pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/checkpoint.msgpack` | pending |

## Observations

- Mandatory inventory selected the exact Kristine-owned v4-8 from the final `AVAILABLE` table.
- Read-only verification found the TPU `READY` and `HEALTHY`, four JAX TPU devices, no VFIO holder or trainer, the node-local central2 dataset present, and W&B credentials available.
- The reviewed change keeps the transformer and message width at 128 and introduces only a 256-wide ReLU hidden layer immediately before the vocabulary projection.

## Interpretation

Pending.

## Updates

- `2026-10-15T05:32:25Z`: The two-pass review returned no findings, commit `9fb9ce1` was pushed to PR 1, and the immutable source snapshot was created. Launch remains pending final remote copy verification and the no-collision preflight.

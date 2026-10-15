# Conlang JAX v4-8 batch-256 2,000-step tied sender-receiver encoders

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100` |
| Git revision | reviewed source and configuration `183b6bd0579142ff8a34a1668fd087d56eef0f02` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders-20261015T151851.174901852Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: true
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
| Sender/receiver memory encoders | independent two-layer encoder stacks | one tied two-layer encoder stack | selected config and `jax_models.py` |

The transformer/message width remains 128, the output hidden width remains 256, and all other scientific hyperparameters, seed, data, optimization, W&B, and top-100 vocabulary-histogram settings match the baseline.

## Results

Pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders.yaml` | yes; SHA-256 `7d7022e8c1bbc6d975e3870817d819f131c2aa74ff97e32c77703f7d44b308b4` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `183b6bd` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders-20261015T151851.174901852Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | local verified; 39 files, 342711 bytes, relative SHA-256-manifest digest `510a882f641b819a58eb19de17711cd978a9def62eb2e0b1aa15f3a38bcd3c51`; remote verification pending |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | previously verified on the selected central2 TPU; fresh verification pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/checkpoint.msgpack` | no; launch pending |
| Tracking run | not recorded | no; launch pending |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- Mandatory inventory selected this exact Kristine-owned v4-8 from the final `AVAILABLE` table; the direct API reported `READY`, `HEALTHY`, Spot, and the occupancy helper reported `AVAILABLE`.
- The implementation stores one shared encoder parameter subtree and preserves the prior independent parameter tree when the option is disabled.
- Shared-encoder path-gradient telemetry isolates sender and receiver uses before reporting their norms.
- Focused tests and the full suite passed on CPU: 106 tests and 7 subtests, with two pre-existing Pydantic warnings.

## Interpretation

- This run tests only encoder weight tying relative to the output-hidden-256 baseline. The receiver decoder and output head remain separate because they have different parameter structures.

## Updates

- `2026-10-15T15:18:51Z`: The two-pass review completed, both confirmed findings were fixed, commit `183b6bd` was pushed to PR 1, and the immutable source snapshot was created. Launch remains pending remote source/data verification and final preflight.

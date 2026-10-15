# Conlang JAX v4-8 batch-256 2,000-step hidden-512 tied encoders

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders` |
| Git revision | reviewed source and configuration `4dca699a171d928edf3e84934b0b1ae96aa8fb4c` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-20261015T155127.637498697Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, expected four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders
wandb_enabled: true
hidden_size: 512
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
| Transformer/message hidden size | 128 | 512 | selected config |

The output hidden dimension remains 256 and the sender/receiver memory encoders remain tied. All data, optimization, seed, training length, W&B, and compact top-100 vocabulary-histogram settings match the baseline.

## Results

Pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders.yaml` | yes; SHA-256 `3b2a4d9b580b2869478df026a48b328e38cd556047c90cdf94976b9cb8821b6d` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `4dca699` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-20261015T155127.637498697Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 40 files and 343676 bytes with relative SHA-256-manifest digest `e70030b02954bf3927a117f9eef38526611f7cc003f1b61bbacc1f6007da27ef` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders/checkpoint.msgpack` | no; launch pending |
| Tracking run | not recorded | no; launch pending |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- Mandatory inventory selected this exact Kristine-owned v4-8 from the final `AVAILABLE` table; the direct API reported `READY`, `HEALTHY`, and Spot, while the occupancy helper reported `AVAILABLE`.
- The model resolves to 16,531,855 parameters, or 66,127,420 bytes of FP32 parameter storage, before optimizer state and activations. This is well within the selected v4-8's model-state capacity; first-step compilation remains the authoritative runtime memory gate.
- The exact YAML passed both sequential PR review passes. The full CPU suite passed: 106 tests and 7 subtests, with two pre-existing Pydantic warnings.
- The Snap-wrapped inventory initially failed locally before any cloud request. The direct packaged Cloud CLI completed the required inventory.
- Two gcloud reconnects later received malformed SSH banners before authentication and before running a remote command. A bounded explicit worker-0 connection using the exact gcloud-generated command verified the same selected TPU and copied the immutable snapshot without changing run identity.

## Interpretation

- This run tests only a fourfold increase in transformer/message hidden size relative to the tied-encoder baseline. Because model geometry changed, the hidden-128 throughput is context rather than a like-for-like performance gate.

## Updates

- `2026-10-15T15:56:47Z`: Configuration commit `4dca699` passed correctness and cleanup review, the full CPU suite passed, the immutable snapshot was verified locally and remotely, and the selected TPU/runtime/regional data passed preflight. Launch remains pending the final occupancy gate.

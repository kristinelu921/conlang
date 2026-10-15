# Conlang JAX v4-8 batch-256 2,000-step hidden-512 tied encoders with role gates

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders` |
| Git revision | reviewed source and configuration `35b35b83774e6104d8e9709307feb276f7e9a54a` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64-20261015T181932.284820169Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, expected four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64
wandb_enabled: true
hidden_size: 512
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
| Separate sender/receiver residual gate width | disabled | 64 channels per role | selected config |

The transformer/message hidden size remains 512, the output hidden dimension remains 256, and the sender/receiver encoder stack remains tied. Data, optimization, seed, training length, and compact top-100 vocabulary-histogram settings match the baseline.

## Results

Pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64.yaml` | yes; SHA-256 `9af1febf0ec9d403890acc81c16ebdc891dd76d620b4e7d0489d22f0dfa2d14f` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `35b35b8` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64-20261015T181932.284820169Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 41 files and 355884 bytes with relative SHA-256-manifest digest `9925550dd13a2929a08b7084979ff5cf5e1c89213354a890bf2be9552ebd8733` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; generated deterministically on the Central2 TPU after the matching US-CENTRAL2 bucket prefix was confirmed absent; 7 files, 11643131 bytes, 15625 worlds, 125000 questions, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Runtime | `/home/kristinelu/elf-runs/runtime/venv` | yes; pinned JAX 0.6.2 profile, freeze SHA-256 `42d44c5882ca2ce9933a0891abf430e9426142616b448939265565d17c2ee18c`, W&B authenticated |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64/checkpoint.msgpack` | no; launch pending |
| Tracking run | not recorded | no; launch pending |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- Mandatory inventory selected this exact Kristine-owned v4-8 from the final `AVAILABLE` table; the direct API reported `READY`, `HEALTHY`, Spot, topology `2x2x1`, and the occupancy helper reported `AVAILABLE`.
- The model resolves to 16,729,743 parameters, 197,888 more than the hidden-512 baseline (+1.197%). Each role gate is an independent bottleneck adapter with sigmoid bias `-2` and a zero-initialized residual output projection, so the treatment starts as the exact ungated function.
- Correctness review found that receiver-gate telemetry initially counted padded question positions. Commit `a199674` changed it to globally weighted masked sums/counts and seven focused tests passed. Cleanup review found stale architecture documentation; commit `35b35b8` documented gate placement and initialization. No material finding remains.
- The final CPU suite passed: 110 tests and 7 subtests, with two pre-existing Pydantic warnings.
- Runtime initialization was scoped to this selected idle TPU. The pinned runtime then passed a fresh no-initialize readiness probe.
- The regional bucket location is `US-CENTRAL2`; its `kristine/datasets/conlangs` prefix was absent, so the established seed-1729 starter dataset was regenerated node-locally and fully verified rather than copied across regions.
- Remote preflight reported 84 GiB free for data and outputs, an absent run output path, and an idle TPU device.

## Interpretation

- This run isolates separate 64-channel sender and receiver residual gates relative to the completed hidden-512 tied-encoder baseline. One seed and a 2,000-step horizon can establish mechanics and a directional result, but not a general quality claim.

## Updates

- `2026-10-15T18:22:32Z`: The reviewed source/config closure was pushed to PR 1, the pinned runtime and deterministic regional data were initialized and verified on the selected TPU, the immutable source snapshot matched locally and remotely, and remote preflight passed. Launch remains pending the final occupancy gate.

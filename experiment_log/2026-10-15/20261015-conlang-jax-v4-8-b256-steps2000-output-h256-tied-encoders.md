# Conlang JAX v4-8 batch-256 2,000-step tied sender-receiver encoders

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders` |
| Status | `completed` |
| Started | `2026-10-15T15:21:17Z` |
| Finished | `2026-10-15T15:26:15Z` |
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

| Metric | Tied encoders | Independent-encoder baseline | Difference |
|---|---:|---:|---:|
| Final training loss | 3.929022 | 3.909020 | +0.020002 |
| Final exact match | 0.0390625 | 0.0390625 | 0.0000000 |
| Final mean joint log-probability | -4.047143 | -4.058864 | +0.011721 |
| Median throughput | 5,868.96 examples/s | 5,843.00 examples/s | +25.97 examples/s (+0.44%) |
| Mean throughput | 5,841.69 examples/s | 5,829.56 examples/s | +12.13 examples/s (+0.21%) |

The tied run contains 200 training records, 20 evaluation records, and 199 post-warmup throughput measurements. Throughput ranged from 4,145.91 to 5,910.02 examples/s. All 200 training records contain finite sender, receiver, shared-encoder gradient/weight path metrics. Every compact vocabulary histogram has at most 22 occupied bins, matching token-ID and edge dimensions under the configured top-100 cap.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders.yaml` | yes; SHA-256 `7d7022e8c1bbc6d975e3870817d819f131c2aa74ff97e32c77703f7d44b308b4` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `183b6bd` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders-20261015T151851.174901852Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 39 files and 342711 bytes with relative SHA-256-manifest digest `510a882f641b819a58eb19de17711cd978a9def62eb2e0b1aa15f3a38bcd3c51` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; 15,625 worlds and 125,000 questions, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/checkpoint.msgpack` | yes; 21,653,009 bytes, SHA-256 `e21a080c3d4e9b7f7a27091a0f079a15c3f06ec641c805810c38bf68220ebd7e`, independently restored at step 2000 on CPU |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/metrics.jsonl` | yes; 220 lines, SHA-256 `82118a0202f3096627f368e5e3ecfadf397e7bcb0936aba8691e3215bbea4ac9` |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/outputs.jsonl` | yes; 200 lines, SHA-256 `694604f6fc4319c883692be4d635b7f3c5b8ee4b34aeb2a4ae27dd02b7ebeda5` |
| Effective configuration | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/config.yml` | yes; SHA-256 `2c79a5054240fed647b7919e9f8d42ad714c043f986842e043acd7457cffb39d` |
| Canonical launcher log | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-tied-encoders/launcher.log.gz` | yes; deterministic gzip SHA-256 `690d957c32c3c694a2f3944334b7e3964678be94ea58be1ba7b85a87eae754fc`; uncompressed remote log SHA-256 `2da40b353b3759f850914486b68b6b6324b659065b099b9e45b9a5588fbd9c5a` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/rffwu89b` | yes; authenticated API reports `finished`, summary step 2000, 200 histogram history rows, no table files, and no token-ID summary |

## Failure Details

- Last known step: 2000, completed
- Error summary: none
- Detailed output: canonical launcher log in the artifacts table

## Observations

- Mandatory inventory selected this exact Kristine-owned v4-8 from the final `AVAILABLE` table; the direct API reported `READY`, `HEALTHY`, Spot, and the occupancy helper reported `AVAILABLE`.
- The implementation stores one shared encoder parameter subtree and preserves the prior independent parameter tree when the option is disabled.
- Shared-encoder path-gradient telemetry isolates sender and receiver uses before reporting their norms.
- Focused tests and the full suite passed on CPU: 106 tests and 7 subtests, with two pre-existing Pydantic warnings.
- JAX reported the TPU backend with four devices before training, and the canonical log contains the exact step-2000 completion sentinel.
- At the terminal audit the TPU remained `READY` and `HEALTHY`; the exact trainer was absent, VFIO had no holders, and the libtpu lockfile was absent.
- W&B stored the requested generated-vocabulary histogram at every training log point. Its public history contains the compact bin payload, while the summary omits token IDs and the run uploaded no table or media files.

## Interpretation

- This run tests only encoder weight tying relative to the output-hidden-256 baseline. The receiver decoder and output head remain separate because they have different parameter structures.
- At this single seed and short horizon, tying the two encoder stacks preserved final exact match, improved mean joint log-probability by 0.0117, and changed median throughput by only +0.44%. The final training loss was 0.0200 higher. These small differences are not evidence of a general quality change without additional seeds or longer training.

## Updates

- `2026-10-15T15:18:51Z`: The two-pass review completed, both confirmed findings were fixed, commit `183b6bd` was pushed to PR 1, and the immutable source snapshot was created. Launch was pending remote source/data verification and final preflight.
- `2026-10-15T15:21:17Z`: W&B run `rffwu89b` started after remote source, runtime, regional data, disk, ownership, and TPU-occupancy checks passed.
- `2026-10-15T15:26:15Z`: Training reached step 2000 and emitted `TRAINING_COMPLETE`; W&B finalized the run as `finished`.
- `2026-10-15T15:29:00Z`: The terminal worker audit found no trainer or VFIO holder, restored the checkpoint at step 2000, verified all remote hashes, and archived the metrics, outputs, effective configuration, and canonical log locally.

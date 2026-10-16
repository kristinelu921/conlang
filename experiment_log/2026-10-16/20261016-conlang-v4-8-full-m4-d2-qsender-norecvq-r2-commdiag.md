# Full-data two-round sender exchange diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag` |
| Status | `running` |
| Started | `2026-10-16T10:40:58Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Git revision | immutable launch snapshot HEAD `9b471b0362605a4b7c0583af4ae2998ebd8735ed`; reviewed implementation/configuration closure `a74ed6fbf1630a625f89c64e026c4b86f0696f5c` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 2000
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
receiver_question_input: false
communication_rounds: 2
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

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Communication rounds | 1 | 2 | Let either sender use the other sender's first-round bridge representation together with its own complementary private hop before the receiver sees the result |

All data, seed, optimization, schedule, widths, depth, message length, sharing, gates, sender question conditioning, receiver question masking, hardware, and causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

The receiver-question bypass and matched wrong-question control establish that the senders can carry the target question, while both message lengths 4 and 8 show no beneficial private-passage gap. The unresolved computation is a two-hop join across passages: the sender holding `person -> middle` can select the bridge entity, but the other sender must combine that bridge with its own `middle -> answer` fact.

Round one remains the existing shared sender applied independently to both question-conditioned passages. In round two, each sender receives the other sender's first-round continuous message alongside its original question and private passage. Both round-two calls are synchronous, reuse exactly the same sender parameters, and expose only their final four-vector messages to the receiver. This covers either assignment of the two hops without using generator-only reasoning metadata.

Keep the second round only if intact final messages beat target-question/wrong-passages messages, ideally in exact match and also in mean joint log probability. Wrong-question messages must remain worse than intact to confirm that question transport survives. If private-passage sensitivity remains flat, the next intervention is an explicit supervised bridge-entity objective rather than more generic width, depth, or weight sharing.

## Results

Training is in progress. The first completed optimizer step passed the launch gate, and steps 20--80 sustained approximately 2,435--2,446 examples/s while loss fell from `7.7973` to `5.3475`.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag.yaml` | parses; correctness and cleanup reviews passed |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed implementation/configuration closure pushed |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag-20261016T103711.365431876Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-r2-commdiag` | local and remote matched before launch: 56 files, 425946 bytes, relative SHA-256-manifest digest `a49cb431a00c35980c0f9ef381b1db54ba8c0e710bd3b564b2879aee16998d05`; selected-config SHA-256 `9289846b850928bed986db8f8be2af6d7b96a8c3df44d45bd9d32b5987d66ad1`; no bytecode caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | revalidated on the TPU: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/daubgk8r` | authenticated, syncing, and reporting training progress |
| Checkpoint | pending | expected at the configured node-local output path |

## Review and Verification

- Focused CPU verification passed 11 cases covering one-round regression behavior, config validation, crossed two-round dependencies in both directions, tied and untied encoders, unchanged parameter trees, and matched wrong-question and wrong-passage controls across both round counts.
- Full CPU regression closure passed 60 tests with two existing Pydantic warnings in 157.54 seconds.
- Correctness and cleanup reviews passed for exact implementation/configuration range `df823a4f7c869be72a6b89b8e49c5b340b3b528b..a74ed6fbf1630a625f89c64e026c4b86f0696f5c`.

## Failure Details

- Last known step: `80` and advancing.
- No failure. The exact trainer owns the TPU lock and all four devices; foreground monitoring remains active through completion.

## Updates

- `2026-10-16`: Selected a synchronous second exchange after the message-8 bypass proved question transport but showed no beneficial private-fact transport.
- `2026-10-16`: Correctness and cleanup reviews passed for the exact implementation/configuration closure; the selected YAML remains the sole run configuration in that closure.
- `2026-10-16T10:37:11Z`: After the exact Kristine v4-8 appeared in the mandatory final `AVAILABLE` table, created immutable launch snapshot `9b471b0362605a4b7c0583af4ae2998ebd8735ed` and verified its 56-file local manifest.
- `2026-10-16T10:39:09Z`: Verified the identical node-local source manifest and config hash, JAX `0.6.2`, the complete region-local dataset, 85 GiB free space, a fresh output path, writable TPU log directory, and idle TPU devices; the selected config parses with two communication rounds.
- `2026-10-16T10:40:58Z`: Launched after the exact same Kristine v4-8 passed the final mandatory `AVAILABLE` audit and launch-boundary path, lock, and VFIO checks; W&B run `daubgk8r` authenticated and synced.
- `2026-10-16`: Passed the first-step launch gate and reached step 80 with stable finite loss and approximately 2,440 examples/s. The expected throughput cost reflects the additional sender pass; no HBM or compilation failure appeared.

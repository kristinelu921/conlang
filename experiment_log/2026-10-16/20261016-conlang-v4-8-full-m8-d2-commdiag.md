# Full-data message-8 depth-2 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m8-d2-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T07:11:43Z` |
| Finished | `2026-10-16T07:17:04Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | snapshot HEAD `45cef137a0b7381a8641d3923e6848b648e76a4f`; reviewed configuration commit `3f1b5c6e1a5e896437197e1bbd9bcd0a5f516491` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-commdiag; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m8-d2-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m8-d2-commdiag.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-commdiag.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m8-d2-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m8-d2-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m8-d2-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
message_length: 8
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
| Message vectors per sender | 4 | 8 | Give each sender one continuous slot per source fact and test channel capacity without changing model width or depth |

All data, optimization, schedule, seed, hidden width, output width, untied weights, disabled gates, depth, precision, hardware, and causal evaluation arms remain fixed.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.0390625` (10/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0525035858` | 2000 | max | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-4.0520439148` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message mean joint log probability | `-4.0525760651` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0525360107` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0525856018` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0527019501` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9147219658` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `5808.5002 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.0024856913` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.0019665654` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.1834628582` | 2000 | diagnostic | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m8-d2-commdiag.yaml` | correctness and cleanup reviews passed at `3f1b5c6e1a5e896437197e1bbd9bcd0a5f516491` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m8-d2-commdiag-20261016T071000.380774403Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m8-d2-commdiag` | yes; local and remote contain 51 files and 377359 bytes; relative SHA-256-manifest digest `9608cdd9ce328435aa975f7b2474658d86ba7bb83c3e0d9ee5a2f63c2844da08` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; same verified Central2 data, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-commdiag/checkpoint.msgpack` | yes; SHA-256 `4b29a7a9a4058d42413936d236670538ba610b58a12c2e05138aedadf9755d81`; CPU-only restore verified step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/1t2nan9n` | synced and finalized by the canonical launcher |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-commdiag/metrics.jsonl` | yes; SHA-256 `4cd499eb77f41152b236262ad25ab27fc70b0a8ecde5c88d190b28d042cbfca9` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-commdiag/outputs.jsonl` | yes; SHA-256 `262e4d272bdd6f0b519f5d0a25b2639b26c768e921c0363ef379895a212875f0` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-commdiag/launcher.log.gz` | yes; compressed SHA-256 `25866bf25bf0be9c22bd58504cfa797ae65c19eeb551e01d6b42c42794ead672`; uncompressed remote SHA-256 `d8e99af1f3387b0b510bafc97fc84b2bb7837c2ad035e0ce28857ea03840b9b2` |

## Failure Details

- Last known step: `2000`
- Error summary: training completed successfully. The first multi-file artifact SCP failed after copying two valid files and one partial output; an isolated retry produced a hash-identical canonical output.
- Detailed output: `experiment_log/2026-10-16/20261016-conlang-v4-8-full-m8-d2-commdiag-artifact-transfer-retry.md`

## Hypothesis and Decision Rule

The message-4 baseline is completely invariant to zero, wrong-world, swapped, and single-sender messages. Because each passage contains eight independently queryable facts but exposes only four learned output slots, channel compression may prevent stable fact-addressable communication. Message length 8 tests that mechanism directly.

Keep message length 8 for later stacked arms only if intact messages gain exact-match or likelihood over zero and wrong-world messages. Ordinary exact match without a causal gap is not evidence for communication.

## Interpretation

- Message length 8 does not create communication. All six causal arms again produce the same 10 correct examples.
- Zero messages improve joint log probability by `0.0004597` relative to intact messages, while wrong-world messages differ by only `-0.0000725`; these are causally flat.
- Sender credit assignment worsened: the receiver gradient norm is approximately 476 times agent 1's and 602 times agent 2's at step 2000.
- The eight-slot channel should not be stacked into later arms based on this evidence. Depth 4 with message length 4 is the next independent test.

## Review and Verification

- Correctness review: pass with no findings. The diff is one YAML; model shape probing produced two `(2, 8, 16)` sender messages and valid `(2, 4, 79)` logits; three focused tests passed.
- Cleanup review: pass. The immutable closure contains exactly one YAML and changes only run identity plus `message_length: 4 -> 8`.
- `TrainConfig.validate()`, the matched-config assertion, and `git diff --check` passed.

## Updates

- `2026-10-16T07:04:00Z`: Planned after the message-4 baseline produced identical `0.0390625` exact match under all six message interventions.
- `2026-10-16T07:11:00Z`: Configuration reviews passed, PR 1 was updated, the mandatory availability audit returned the same Kristine v4-8 in its final `AVAILABLE` table, the immutable node-local snapshot matched locally and remotely, and preflight reported 86 GiB free with VFIO idle.
- `2026-10-16T07:11:43Z`: Launched after an immediate final occupancy gate returned `AVAILABLE`; W&B run `1t2nan9n` authenticated and began syncing.
- `2026-10-16T07:17:04Z`: Completed step 2000 and all causal arms. The canonical log emitted `TRAINING_COMPLETE`, W&B finalized, checkpoint restore passed, no run process remained, VFIO was idle, and the TPU remained `READY` / `HEALTHY`.

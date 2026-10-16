# Full-data sender-only question-path communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T08:49:14Z` |
| Finished | `2026-10-16T08:56:27Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-commdiag` |
| Git revision | immutable snapshot HEAD `9772ded06ab02cdf00edd7239405c59ed6d92cbd`; reviewed implementation/config closure `bc4630f78867f1983fad2456da6a44a33ed0c333` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
receiver_question_input: false
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
|---|---|---|---|
| Direct receiver question input | enabled | disabled | Remove the confirmed cheap question-only path so the question and passage evidence must reach the receiver through sender messages |

All data, seed, optimization, schedule, widths, message length, depth, sharing, gates, sender inputs, hardware, and causal evaluation arms remained fixed.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.03515625` (9/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0549392700` | 2000 | max | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-10.2607593536` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0552206039` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0550308228` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0554199219` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0555276871` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9214811325` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `4450.5006 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.0233914331` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.0192575324` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.1896870136` | 2000 | diagnostic | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-commdiag.yaml` | correctness and cleanup reviews passed |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | linked; completed results recorded in this report |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag-20261016T084658.016087401Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` | yes; local and remote contain 54 files and 412309 bytes; relative SHA-256-manifest digest `9f967e6d0ca2cd6a17e77b0023dae6fe1818872a4008d4692dd96f4840d948eb` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; 15,625 worlds / 125,000 questions; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag/checkpoint.msgpack` | yes; 26,418,137 bytes; SHA-256 `74508f61219457ca5dc3466772d78ca484caeb954855458aa3f9bf4bafa66c68`; CPU-only restore verified step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/suzue83n` | synced and finalized by the canonical launcher |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag/metrics.jsonl` | yes; 344,018 bytes; SHA-256 `d49e53b6f4b35b8c78b902d5d235dba0d5ebd49abce8e1cfa4dfc7994459c105` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag/outputs.jsonl` | yes; 2,812,407 bytes; SHA-256 `9e44aaf7b965cb190a2abde7909ba3be9dba62c133c996c671869756c9cecca9` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag/launcher.log.gz` | yes; compressed SHA-256 `2fe4d352450c1e882767e0b32284f2079f346b7bfe159f2a4b3e4ca1911fe1cd`; uncompressed remote SHA-256 `1917d99c99d2e4544d6ab5866d548d00aa86733f3ab16f301f21b86d8e5c3d05` |

## Hypothesis and Decision Rule

Question-conditioned senders did not produce a beneficial passage-specific causal gap while the receiver still saw the question directly. That direct path could continue fitting relation-conditioned answer priors without using messages. This arm masked all receiver question positions; the senders remained question-conditioned, so the question and private evidence could reach the answer decoder only through their eight combined message vectors.

Useful communication required intact messages to outperform both zero messages and target-question/wrong-passages messages. A large intact-versus-zero gap without a meaningful intact-versus-wrong-passages gap would show question transport but not useful private-content transport.

## Interpretation

- The messages now carry indispensable question information: zeroing them drops exact match from 9/256 to 2/256 and mean joint log probability by `6.20582`.
- Private passage evidence is still not useful at the task level. Target-question/wrong-passages preserves 9/256 and changes mean joint log probability by only `-0.0002813`; this tiny favorable sign for intact passages is not a convincing or practically useful private-content gap in a single 256-example evaluation.
- Sender credit assignment improves sharply. The receiver gradient is about 51 times agent 1's and 62 times agent 2's, versus about 476--673 times in the matched direct-question arm. This supports the direct receiver question path as a major source of sender gradient starvation.
- Exact accuracy remains at the 3.5% question-prior level. The architecture has learned to relay the question through messages, not to bind the two private hops into an answer.
- Because messages are now causally active, channel capacity can be retested as an interaction: message length 8 may give each mixed passage enough slots to preserve its four answer-side mappings plus question-selected evidence. If that remains passage-flat, the next intervention should directly supervise intermediate content or add sequential cross-agent exchange.

## Failure Details

- Last known step: `2000`
- Error summary: training and evaluation completed successfully. Artifact SCP retried transient SSH MAC errors automatically; every local artifact hash matches the remote source.
- Terminal evidence: canonical single-worker log contains `TRAINING_COMPLETE ... step=2000`; W&B finalized; no exact trainer remained; VFIO was idle; TPU remained `READY` / `HEALTHY`.

## Review and Verification

- Correctness review passed for both teacher-forced and greedy masking, sender conditioning, target-question/wrong-passages semantics, tied/untied compatibility, default parameter-tree compatibility, and the causal decision rule.
- Cleanup review passed after receiver documentation was aligned with optional question context.
- The full CPU regression suite passed 52 tests with two existing Pydantic warnings.
- The selected YAML validated and differed from the completed question-conditioned arm only by fresh run identity fields and `receiver_question_input: false`.
- Preflight reported 85 GiB free with TPU devices idle. The exact device appeared in the final mandatory `AVAILABLE` table immediately before launch.

## Updates

- `2026-10-16T08:46:58Z`: Created the immutable reviewed snapshot and verified an identical 54-file source manifest locally and remotely.
- `2026-10-16T08:49:14Z`: Launched after the exact same Kristine v4-8 passed the final availability/occupancy gate; W&B run `suzue83n` authenticated and synced.
- `2026-10-16T08:52:45Z`: Initial TPU compilation completed; training began around 4,450 examples/s.
- `2026-10-16T08:56:27Z`: Completed step 2000, all causal arms, checkpoint save, and W&B finalization.
- `2026-10-16T08:56:35Z`: CPU restore verified checkpoint step 2000; process/VFIO checks were idle and TPU state remained `READY` / `HEALTHY`.

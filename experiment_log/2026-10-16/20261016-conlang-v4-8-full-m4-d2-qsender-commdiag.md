# Full-data question-conditioned sender communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T08:16:53Z` |
| Finished | `2026-10-16T08:24:51Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | immutable snapshot HEAD `4fd01a41561c372527078d2c14948da1a15a166b`; reviewed implementation/config closure `7a45697391d2d36ba0ec84af37af33f0460f0131` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qsender-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qsender-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
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
| Sender input | private passage only | question followed by private passage | Shorten the credit path by letting each sender select question-relevant evidence before compressing its passage into four vectors |

All data, seed, optimization, schedule, widths, message length, depth, weight sharing, gates, receiver inputs, hardware, and causal evaluation arms remained fixed.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.0390625` (10/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0559148788` | 2000 | max | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-4.0558824539` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0557312965` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0558319092` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0553860664` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.04296875` (11/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0554971695` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9144630432` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `4462.3184 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.0024527854` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.0017344982` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.1678500175` | 2000 | diagnostic | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-commdiag.yaml` | correctness and cleanup reviews passed |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; result update pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag-20261016T081332.874739292Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag` | yes; local and remote contain 53 files and 407094 bytes; relative SHA-256-manifest digest `cb14af919c89ca14348cd333371975065657a4d7b1ac559674424caa36b68ee0` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; 15,625 worlds / 125,000 questions; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag/checkpoint.msgpack` | yes; 26,418,137 bytes; SHA-256 `2f69b5bd6a558e6c04c06e3c0694c657bf6f9116457e089dd367bb0ecb5e21fa`; CPU-only restore verified step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/mchumqhp` | synced and finalized by the canonical launcher |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag/metrics.jsonl` | yes; 346,568 bytes; SHA-256 `5687d6ee8574434ea73ab1c37351033623666c57feb481e8e0fb20da3bd68fb8` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag/outputs.jsonl` | yes; 2,806,369 bytes; SHA-256 `c0f0cbcfd55b353007d98f11a268ae2a2eefb8596cb04d57ee8f0065ce8eb6a1` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-commdiag/launcher.log.gz` | yes; compressed SHA-256 `2e250fed03b5d5eda2a596914e46bd59c2218914d22118b14aed11cb69d2f9ae`; uncompressed remote SHA-256 `44cb8461060474794f3699360594834bbd1cf1f8171698b3556bfb3f3ca9a49a` |

## Hypothesis and Decision Rule

The passage-only sender must learn a query-independent code for eight facts using only a weak gradient routed through the receiver, while the receiver can cheaply fit question-conditioned answer priors. Conditioning both senders on the question was intended to make the four message vectors query-specific and strengthen sender learning without increasing width, depth, or channel length.

Private-content communication required intact messages to outperform the target-question/wrong-passages control in exact match or joint log probability, with zero messages as a second reference. The wrong-world arm rotated only the two private passages and re-encoded them against each target row's unchanged question.

## Interpretation

- Question-conditioned senders still do not use private passage content. Intact and target-question/wrong-passages both produce 10/256, and wrong passages improve mean joint log probability by `0.0001836`.
- Zero messages lose one exact answer but improve likelihood by `0.0000324`; agent 2 alone gains one answer and improves likelihood by `0.0004177`. These directions do not support a useful two-sender channel.
- Sender credit remains extremely weak: the receiver gradient is about 476 times agent 1's and 673 times agent 2's at step 2000, worse than the passage-only baseline's roughly 210--251-fold imbalance.
- The extra question context lowers throughput from about 5,870 to 4,462 examples/s without creating a passage-specific causal gap. Do not stack this switch into width, depth, or weight-sharing arms.
- The long-credit-path hypothesis is now insufficient by itself. The next intervention should directly supervise or contrast sender content rather than relying only on downstream answer likelihood.

## Failure Details

- Last known step: `2000`
- Error summary: training and evaluation completed successfully. Artifact SCP encountered a transient SSH MAC error and recovered automatically; every local artifact hash matches the remote source.
- Terminal evidence: canonical single-worker log contains `TRAINING_COMPLETE ... step=2000`; W&B finalized; no exact trainer remained; VFIO was idle; TPU remained `READY` / `HEALTHY`.

## Review and Verification

- Correctness review found and fixed a pre-launch causal-control confound: conditioned wrong-world evaluation now rotates only private passages and retains each target question in both teacher-forced and greedy paths.
- Cleanup review passed after sender documentation was aligned with optional question context.
- The final CPU regression suite passed 50 tests with two existing Pydantic warnings. Focused tied/untied wrong-world oracle coverage passed both cases.
- The selected YAML validated and differed from the causal baseline only by fresh run identity fields and `sender_question_conditioning: true`.
- Preflight reported 86 GiB free with TPU devices idle. The exact device appeared in the final mandatory `AVAILABLE` table immediately before launch.

## Updates

- `2026-10-16T08:13:32Z`: Created the immutable reviewed snapshot and verified an identical 53-file source manifest locally and remotely.
- `2026-10-16T08:16:53Z`: Launched after the exact same Kristine v4-8 passed the final availability/occupancy gate; W&B run `mchumqhp` authenticated and synced.
- `2026-10-16T08:20:29Z`: Initial TPU compilation completed; training began around 4,460 examples/s.
- `2026-10-16T08:24:51Z`: Completed step 2000, all causal arms, checkpoint save, and W&B finalization.
- `2026-10-16T08:25:00Z`: CPU restore verified checkpoint step 2000; process/VFIO checks were idle and TPU state remained `READY` / `HEALTHY`.

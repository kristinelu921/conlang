# Full-data message-8 depth-2 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m8-d2-commdiag` |
| Status | `planned` |
| Started | `not recorded` |
| Finished | `not recorded` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | reviewed configuration commit `3f1b5c6e1a5e896437197e1bbd9bcd0a5f516491`; snapshot HEAD pending evidence commit |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m8-d2-commdiag.yaml` inside the immutable TPU-local snapshot; complete launcher pending staging |
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
| Validation | Intact-message exact match | `not recorded` | 2000 | max | pending |
| Validation | Zero-message exact match | `not recorded` | 2000 | diagnostic | pending |
| Validation | Wrong-world-message exact match | `not recorded` | 2000 | diagnostic | pending |
| Validation | Swapped-sender exact match | `not recorded` | 2000 | diagnostic | pending |
| Validation | Agent-1-only exact match | `not recorded` | 2000 | diagnostic | pending |
| Validation | Agent-2-only exact match | `not recorded` | 2000 | diagnostic | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m8-d2-commdiag.yaml` | correctness and cleanup reviews passed at `3f1b5c6e1a5e896437197e1bbd9bcd0a5f516491` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `not recorded` | pending |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | established Central2 copy; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-commdiag/checkpoint.msgpack` | pending |
| Tracking run | `not recorded` | pending |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-commdiag/metrics.jsonl` | pending |

## Failure Details

- Last known step: `not recorded`
- Error summary: not applicable; launch pending
- Detailed output: `not recorded`

## Hypothesis and Decision Rule

The message-4 baseline is completely invariant to zero, wrong-world, swapped, and single-sender messages. Because each passage contains eight independently queryable facts but exposes only four learned output slots, channel compression may prevent stable fact-addressable communication. Message length 8 tests that mechanism directly.

Keep message length 8 for later stacked arms only if intact messages gain exact-match or likelihood over zero and wrong-world messages. Ordinary exact match without a causal gap is not evidence for communication.

## Review and Verification

- Correctness review: pass with no findings. The diff is one YAML; model shape probing produced two `(2, 8, 16)` sender messages and valid `(2, 4, 79)` logits; three focused tests passed.
- Cleanup review: pass. The immutable closure contains exactly one YAML and changes only run identity plus `message_length: 4 -> 8`.
- `TrainConfig.validate()`, the matched-config assertion, and `git diff --check` passed.

## Updates

- `2026-10-16T07:04:00Z`: Planned after the message-4 baseline produced identical `0.0390625` exact match under all six message interventions.

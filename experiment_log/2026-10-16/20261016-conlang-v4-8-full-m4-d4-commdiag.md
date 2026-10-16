# Full-data message-4 depth-4 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d4-commdiag` |
| Status | `planned` |
| Started | `not recorded` |
| Finished | `not recorded` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | reviewed configuration commit `52c47a8bd817f63c586d0deea87aea25784fe0df`; snapshot HEAD pending evidence commit |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml` inside the immutable TPU-local snapshot; complete launcher pending staging |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d4-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d4-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
message_length: 4
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 4
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Transformer layers per stack | 2 | 4 | Test whether character parsing, fact selection, and the two-hop join need more sequential computation |

Every other data, optimization, schedule, seed, width, message-length, sharing, gating, precision, hardware, and causal-evaluation setting remains fixed.

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
| Configuration | `configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml` | correctness and cleanup reviews passed at `52c47a8bd817f63c586d0deea87aea25784fe0df` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `not recorded` | pending |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | established Central2 copy; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag/checkpoint.msgpack` | pending |
| Tracking run | `not recorded` | pending |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d4-commdiag/metrics.jsonl` | pending |

## Failure Details

- Last known step: `not recorded`
- Error summary: not applicable; launch pending
- Detailed output: `not recorded`

## Hypothesis and Decision Rule

Message length 8 did not create any causal message dependence, ruling out four slots as the primary bottleneck. The remaining capacity hypothesis is that two layers per sender/receiver stack are insufficient for parsing, selection, and composition. This arm returns to message length 4 and doubles only depth.

Keep depth 4 for later stacking or width/sharing comparisons only if intact messages outperform zero or wrong-world messages in exact match or answer likelihood. If it stays causally flat, the next intervention should target optimization/credit assignment rather than stack capacity.

## Review and Verification

- Correctness review: pass with no findings. `num_layers` reaches all sender-encoder, receiver-encoder, and decoder loops; a CPU forward pass confirmed four layers, sender messages `(2, 4, 128)`, and logits `(2, 4, 81)`.
- Cleanup review: pass. The immutable closure is one YAML and changes only run identity plus `num_layers: 2 -> 4`.
- `TrainConfig.validate()`, matched-config assertion, three focused tests, and `git diff --check` passed.
- The model has 3,523,599 parameters; reviewer estimates and the prior larger hidden-512 run show no likely v4-8 HBM blocker, with TPU compilation retained as the authoritative gate.

## Updates

- `2026-10-16T07:23:00Z`: Planned after the matched message-8 arm remained causally flat and reduced sender-path gradient norms.

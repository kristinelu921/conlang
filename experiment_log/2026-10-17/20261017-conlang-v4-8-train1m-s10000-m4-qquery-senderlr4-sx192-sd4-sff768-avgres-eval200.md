# Capacity-matched shared sender with an average residual

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag` |
| Git revision | pending immutable launch snapshot |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200.yaml` |
| Command | not recorded; pending immutable snapshot and final launch gate |
| Seed | `1730` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; current API state READY/HEALTHY and occupancy AVAILABLE |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1-train1m
dataset_prefix: kristine/datasets/conlangs-train1m
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 4.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 10000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
sender_hidden_size: 192
sender_num_layers: 4
sender_num_heads: 6
sender_feedforward_size: 768
sender_average_residual: true
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: false
receiver_question_input: true
receiver_question_query_only: true
communication_rounds: 1
bridge_supervision_weight: 0.0
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
| Sender hidden width | `128` | `192` | selected configuration |
| Sender encoder depth | `2` | `4` | selected configuration |
| Sender attention heads | `4` | `6` | selected configuration |
| Sender feed-forward width | `512` | `768` | selected configuration |
| Sender average residual | disabled | enabled | selected configuration |
| Sender parameters | `397,312` | `1,830,464` | initialized production Flax parameter tree |

Run identity, output path, and W&B display name are fresh. Receiver dimensions,
the four-by-128 message channel, optimizer, sender learning-rate multiplier,
seed, batch, precision, regional dataset, 10,000-step horizon, fixed validation
prefix, 200-step intervention cadence, and all causal controls remain matched.
Both passage agents reuse the same sender parameters. This is intentionally a
joint sender-capacity and initialization intervention; a positive result cannot
separate those two mechanisms without a later ablation.

## Hypothesis and Decision Rule

The matched baseline learns a sharp sender-gradient pulse near step 830 but no
stable passage-specific communication. The larger shared sender has four
192-wide layers with six 32-dimensional heads and a TPU-aligned 768-wide FFN.
It projects back to the unchanged 128-dimensional message channel. Four masked
contiguous passage-region averages provide the initial messages exactly; the
learned sender projection begins at zero and can acquire a residual correction
without any auxiliary target or question input.

The treatment succeeds only if intact messages acquire a reproducible advantage
over wrong-world messages across multiple scheduled evaluations, preferably
with an exact-match improvement. Intact-versus-zero alone remains an
out-of-distribution carrier-dependence diagnostic. Compare the step-830 region,
sender/receiver gradient and relative-update ratios, message variation,
single-sender arms, output collapse, and final validation against the baseline.
Because width/depth and the average residual change together, attribute a
positive result to the combined sender treatment rather than either component.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200.yaml` | pending immutable snapshot |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR; update pending |
| Source snapshot | not recorded | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | pending selected-worker verification |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200/checkpoint.msgpack` | no |
| Metrics | pending run-scoped artifact directory | no |
| Outputs | pending run-scoped artifact directory | no |
| Launcher log | pending run-scoped artifact directory | no |
| Tracking run | not recorded | no |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The required filtered inventory selected an idle region-matched Central2
  v4-8 from its final `AVAILABLE` table. Direct describe reports READY, HEALTHY,
  Spot `v4-8`; the read-only occupancy helper reports AVAILABLE.
- The baseline's receiver has `1,720,847` parameters. The selected sender target
  has `1,830,464` parameters, 6.37% more than the unchanged receiver. The full
  initialized model has `3,831,759` FP32 parameters (`15,327,036` raw bytes).
  The sender tree contains one `(4, 192)` query table, `(128, 192)` input and
  `(192, 128)` output projections, and four `(192, 768)` feed-forward inputs.

## Interpretation

- Pending results. A lower loss without a durable intact-versus-wrong-world gap
  would mean greater sender capacity or the average bypass improves optimization
  without establishing useful private-content transport.

## Review and Verification

- The selected YAML parses and validates through `TrainConfig`. Relative to the
  completed eval-200 baseline, its only scientific changes are the five listed
  sender capacity/initialization fields; run, output, and W&B identities are
  fresh.
- Production initialization verified `3,831,759` total parameters with exactly
  one `1,830,464`-parameter sender tree shared across both passage calls.
- Focused CPU verification passed 9 sender, initialization, sharing, shape, and
  causal-intervention tests in `46.91s`.
- The full CPU suite passed `142` tests and `7` subtests with two existing
  Pydantic warnings in `204.71s` under `USE_TORCH_XLA=0` and
  `JAX_PLATFORMS=cpu`.
- Correctness review found no confirmed issue in immutable range
  `930cc28689ad3d4a2d3b24fb487b570a2a140db4..e236107a0e0cd814e9b32c06c2ad9bfdded29701`.
  Cleanup review is pending the updated closure.

## Updates

- `2026-10-17`: Planned the jointly larger and average-initialized shared sender
  treatment, preserving receiver and channel dimensions and the matched
  10,000-step/evaluation protocol.

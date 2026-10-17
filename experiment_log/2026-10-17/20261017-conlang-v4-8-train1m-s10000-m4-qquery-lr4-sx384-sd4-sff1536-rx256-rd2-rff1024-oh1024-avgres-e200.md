# Approximately 4x sender and receiver capacity

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Git revision | `97d0c9256d639b21bed05b978f2040eec4df99fa` (configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml` |
| Command | not recorded; pending immutable snapshot |
| Seed | `1730` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; current API state READY/HEALTHY and occupancy AVAILABLE |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200
wandb_enabled: true
hidden_size: 256
output_hidden_size: 1024
sender_hidden_size: 384
sender_num_layers: 4
sender_num_heads: 12
sender_feedforward_size: 1536
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
num_heads: 8
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Receiver hidden width | `128` | `256` | selected configuration |
| Receiver attention heads | `4` | `8` | selected configuration |
| Receiver feed-forward width | `512` | `1024` | derived as `4 * hidden_size` |
| Output-hidden width | `256` | `1024` | selected configuration |
| Sender hidden width | `192` | `384` | selected configuration |
| Sender attention heads | `6` | `12` | selected configuration |
| Sender feed-forward width | `768` | `1536` | selected configuration |
| Message vector width | `128` | `256` | coupled to receiver hidden width |
| Sender parameters | `1,830,464` | `7,298,176` (`3.987x`) | initialized production Flax trees |
| Receiver parameters | `1,720,847` | `6,986,639` (`4.060x`) | initialized production Flax trees |
| Total parameters | `3,831,759` | `14,845,711` (`3.874x`) | initialized production Flax trees |

Run identity, output path, and W&B display name are fresh. All other scientific
and operational fields remain matched: sender depth 4,
receiver depth 2, four message slots per sender, average-passage residual,
question-blind shared sender, receiver question-only query path, sender learning
rate multiplier 4, optimizer, seed, 1M-example Central2 dataset, global batch
256, bfloat16 precision, 10,000 steps, fixed 256-example validation prefix,
and 200-step causal evaluation cadence.

## Hypothesis and Decision Rule

The previous treatment removed raw sender-gradient starvation but did not learn
passage-specific communication. This arm tests whether approximately four times
as many parameters in each sender and receiver, plus the naturally wider
message carrier and output head, lets answer-token cross-entropy discover the
two-hop communication circuit without auxiliary labels.

The treatment succeeds only if intact messages develop a repeatable advantage
over wrong-world messages across multiple scheduled evaluations, preferably in
exact match. Intact-versus-zero remains only an out-of-distribution learned-
carrier diagnostic. Compare the intact/wrong-world gap trajectory, final-50
sender/receiver gradients and relative updates, message RMS, output diversity,
validation stability, and throughput with the exact baseline. Because receiver,
sender, message, and output widths change together, any improvement belongs to
the coupled scaling treatment and cannot identify one component as causal.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml` | parsed, validated, and CPU-initialized; committed at `97d0c9256d639b21bed05b978f2040eec4df99fa`; PR publication pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR; update pending review |
| Source snapshot | not recorded; pending staging | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | prior verified Central2 copy exists; fresh integrity check pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200/checkpoint.msgpack` | no |
| Metrics | pending run-scoped artifact directory | no |
| Outputs | pending run-scoped artifact directory | no |
| Launcher log | pending run-scoped artifact directory | no |
| Tracking run | not recorded; pending W&B initialization | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- CPU production initialization with vocabulary size `2191` gives `14,845,711`
  FP32 parameters: `560,896` shared embedding, `7,298,176` shared sender, and
  `6,986,639` receiver.
- Sender and receiver attention retain 32-dimensional heads. All selected
  hidden, feed-forward, output, and per-head dimensions are multiples of 32;
  feed-forward and output widths are multiples of 128.
- The required filtered inventory returned five AVAILABLE Kristine TPUs. The
  selected Central2 v4-8 already holds the verified regional dataset and runtime
  from the matched baseline. A direct describe reports READY/HEALTHY, and the
  read-only occupancy helper reports AVAILABLE.

## Interpretation

- Pending results. A lower loss or larger message magnitude without a durable
  intact-versus-wrong-world gap would mean additional capacity still optimizes
  a non-passage-specific carrier or answer prior rather than communication.

## Review and Verification

- Preliminary CPU validation: the YAML parses through `TrainConfig`, all head
  divisibility and node-local path constraints pass, and the production model
  initializes with the stated parameter counts and a 4-by-256 message from each
  sender.
- Correctness review found no confirmed issue in immutable range
  `708eee6b258df8e9f9049087ad0490b88d8693ae..97d0c9256d639b21bed05b978f2040eec4df99fa`.
  It independently verified the YAML delta, production shapes and parameter
  counts, finite forward and greedy paths, seven focused tests, static
  per-device batch, HBM plausibility, causal rule, scope, and secrets. Cleanup
  review found only stale commit-state wording and an overbroad matched-fields
  sentence; this follow-up records the immutable commit and explicitly excludes
  the three fresh identity/path fields.

## Updates

- `2026-10-17T19:43:50Z`: Selected the approximately 4x-capacity treatment,
  confirmed current filtered Kristine TPU availability, and created the planned
  report. No TPU state was changed and no training process was started.

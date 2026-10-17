# T5-tiny warm start with 2x base learning rate and 3x scale

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `planned` |
| Started | `not recorded` |
| Finished | `not recorded` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Git revision | `e38af6e2f3e6761943ed786b32c0737ebf1462aa` (configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` |
| Command | `not recorded; pending immutable snapshot and final launch preflight` |
| Seed | `1730` |
| Hardware | selected candidate `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; final preflight pending |

## Configuration

Source: `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1-train3m
dataset_prefix: kristine/datasets/conlangs-train3m
batch_size: 256
learning_rate: 0.0006
sender_learning_rate_multiplier: 4.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 30000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack
pretrained_initialization: google_t5_efficient_tiny
pretrained_model_dir: /home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8
pretrained_checkpoint_sha256: 16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26
pretrained_tokenizer_sha256: 88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
wandb_enabled: true
hidden_size: 256
output_hidden_size: 256
sender_hidden_size: 256
sender_num_layers: 4
sender_num_heads: 4
sender_feedforward_size: 1024
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
num_layers: 4
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Completed baseline | Current | Source |
|---|---:|---:|---|
| Base learning rate | `0.0003` | `0.0006` | selected configurations |
| Training questions | `1,000,000` | `3,000,000` | deterministic dataset manifests |
| Training worlds | `125,000` | `375,000` | 8 questions per world |
| Training steps | `10,000` | `30,000` | selected configurations |
| Training presentations | `2,560,000` | `7,680,000` | steps x global batch 256 |
| Nominal passes over training questions | `2.56` | `2.56` | presentations / training questions |
| Scheduled causal evaluations | `50` | `150` | unchanged 200-step cadence |

Architecture, T5 checkpoint initialization, global batch, sender learning-rate
multiplier, optimizer, gradient clipping, seed, question/message routing, and
evaluation prefix remain matched. The treatment changes learning rate, data
size, and duration together as requested, so it cannot isolate any one change.

## Hypothesis and Decision Rule

The run asks whether the warm-started model's strong sender credit becomes
useful when given three times as much distinct training data, three times as
many optimization steps, and twice the base learning rate while preserving the
same 2.56 nominal data passes. Success still requires a repeatable positive
intact-over-wrong-world validation gap, preferably with improving exact match;
lower teacher-forced loss alone is insufficient. The 150 evaluations will also
show whether any communication transition is persistent rather than a single
loss discontinuity.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` | pending parse and validation |
| Immutable source | `not recorded; pending staging` | no |
| Source model | `/home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8` | pending selected-node hash verification |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train3m` | pending generation or regional reuse and full verification |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack` | no; pending launch |
| Tracking run | `not recorded` | no |

## Failure Details

- Last known step: not applicable; not launched.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The requested scaling preserves data exposure in epochs: both arms present
  each training example 2.56 times on average.
- The completed baseline sustained a last-50 median `2,143.996` examples/s. At
  that rate, 30,000 training steps contain about 59.7 minutes of steady-state
  work, plus compilation and 150 intervention evaluations.
- The required filtered inventory selected only `kristine` resources; the
  Central2 v4-8 candidate is preferred because it is the proven topology for
  this exact model and retains region-local predecessor data/model artifacts.

## Interpretation

- The doubled base learning rate is a meaningful stability intervention. The
  global norm clip remains one, but pre-clipping spikes can still change AdamW
  dynamics; finite early steps and the causal curves must be checked before
  interpreting the longer run.

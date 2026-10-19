# Tiny 32-dimensional single-link communication gate retry 2

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
| Predecessor | retry 1 failed during source transfer before copying a file |
| Git revision | `d2c3bc74df162b6545937935b8cf5c5ae24ea4d7` (exact configuration commit) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml` |
| Command | pending fresh archive staging and final remote gate |
| Seed | `1729` |
| Hardware | selected `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; READY/HEALTHY and idle at retry selection |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-v1
dataset_prefix: kristine/datasets/conlangs-single-link-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 200
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2
wandb_enabled: true
hidden_size: 32
output_hidden_size: null
sender_hidden_size: 32
sender_num_layers: 2
sender_num_heads: 4
sender_feedforward_size: 128
sender_average_residual: false
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: false
receiver_question_input: true
receiver_question_query_only: false
communication_rounds: 1
bridge_supervision_weight: 0.0
message_length: 1
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

No scientific hyperparameter changed from retry 1. Only run, output, W&B, and
immutable staging identities change. Retry 2 transfers a SHA-256-recorded
regular-file archive rather than passing a `/.` recursive source to SCP.

Relative to operational throughput baseline
`20261014-conlang-jax-v5e8-b256-opt1`, both sender and receiver shrink from
width 128 to 32, each sender emits one vector instead of four, feed-forward
width shrinks from 512 to 128, and each sender sees one required fact instead
of eight mixed facts. Depth remains 2 on both sides.

## Gate Decision Rule

The gate must initialize on all TPU devices, complete 200 finite steps, save a
restorable checkpoint, and produce the scheduled validation interventions.
Promotion to a fresh full-run identity additionally requires no unexplained
material steady-state throughput regression.

## Results

No training result yet. The deterministic dataset contains 100,000 training
questions and 25,000 validation/test questions across 15,625 worlds.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml` | parses and validates; SHA-256 `85973ed1985e439aa878f657b5c30af3d3bbe468b1c5be35176892c772f3c1f3` |
| Dataset manifest | `datasets/linked_facts_v1/manifest.json` | full streaming verifier passed; SHA-256 `5f95f8f4f25a10d12704959c6f275f978548b3641a4db03338d76d4f8d654a13` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1` | seven objects / 11,992,785 bytes; bucket location `US-WEST4` |
| Immutable source | pending fresh snapshot and archive | no |
| Checkpoint | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/checkpoint.msgpack` | no |
| Tracking run | pending | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable to retry 2; both predecessor staging failures
  have distinct immutable reports.
- Detailed output: not recorded.

## Observations

- The selected candidate remained READY, HEALTHY, Spot, and VFIO idle.
- Remote runtime imports passed with JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`,
  and W&B `0.28.1`.
- Preflight reported 4 GiB node-local free space, safely above the 12 MB
  dataset, sub-megabyte source, and small-model checkpoint/log budget.
- The exact model has 228,271 parameters and emits message shape
  `(batch, 2, 32)`: one vector from each sender.

## Interpretation

- Retry 2 is scientifically identical to its predecessors. The fresh identity
  preserves both zero-step staging failures while using a transfer method with
  an explicit archive hash and extracted file-manifest verification.

## Updates

- `2026-10-19T02:46:31Z`: Selected retry 2 after preserving retry 1's empty
  remote target and repeated deterministic SCP error; exact config parsed.

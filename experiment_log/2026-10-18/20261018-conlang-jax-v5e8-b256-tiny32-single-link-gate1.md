# Tiny 32-dimensional single-link communication gate

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
| Baseline | `20261014-conlang-jax-v5e8-b256-opt1` for operational throughput only; architecture and data contract differ |
| Git revision | `117c839bbda5dffb992f7f75af21ca2cc01ab571` (exact configuration commit); source/data commit `b06656e5621729f762e10543703e32ea56aa12e7` |
| Pull request | pending |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1.yaml` |
| Command | pending immutable staging and remote preflight |
| Seed | `1729` |
| Hardware | selected candidate `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, `v5litepod-8`; READY/HEALTHY and idle at selection |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1
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
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1
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

| Parameter | Operational baseline | Current | Source |
|---|---:|---:|---|
| Sender input | 8 mixed world facts | 1 question-required fact | deterministic schema-4 manifest |
| Receiver hidden size | `128` | `32` | selected configurations |
| Sender hidden size | `128` | `32` | selected configurations |
| Continuous vectors per sender | `4` | `1` | selected configurations |
| Sender feed-forward size | `512` | `128` | selected configurations |
| Sender layers | `2` | `2` | selected configurations |
| Receiver encoder/decoder layers | `2` | `2` | selected configurations |
| Model parameters | `1,886,863` | `228,271` | CPU initialization of the exact model trees |

Global batch 256, base learning rate `3e-4`, AdamW settings, seed 1729,
200-step gate duration, and bfloat16 TPU compute remain matched to the
operational baseline. The changed data contract makes this a new scientific
treatment rather than a controlled throughput comparison.

## Gate Decision Rule

The gate must initialize on all TPU devices, complete 200 finite steps, save a
restorable checkpoint, and produce the scheduled validation interventions.
Promotion to a fresh full-run identity additionally requires no material
steady-state throughput regression unexplained by evaluation or logging. Exact
match and intact-versus-wrong-world likelihood are recorded but are not used to
retroactively modify this gate.

## Results

No training result yet. The deterministic dataset contains 100,000 training
questions and 25,000 validation/test questions across 15,625 worlds.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1.yaml` | parses and validates; SHA-256 `2e5973eefe84fbfcb4fd512f4d32846b101eb3bfeb49a5dd43c163803589d7bc` |
| Dataset manifest | `datasets/linked_facts_v1/manifest.json` | full streaming verifier passed; SHA-256 `5f95f8f4f25a10d12704959c6f275f978548b3641a4db03338d76d4f8d654a13` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1` | pending upload and remote verification |
| Immutable source | pending staging under `/kmh-nfs-ssd-us-mount/staging/kristine/elf/` | no |
| Checkpoint | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1/checkpoint.msgpack` | no |
| Tracking run | pending | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The required filtered TPU inventory selected only a `kristine` resource.
- The selected v5e-8 candidate was READY, HEALTHY, and had no VFIO holder at
  selection time.
- The exact model has 70,112 shared-embedding parameters, 25,696 sender
  parameters, 132,463 receiver parameters, and 228,271 parameters total.
- Generator, loader, model-shape, and regional-path tests pass after rebuilding
  the committed dataset with schema version 4.

## Interpretation

- A single 32-dimensional vector is emitted by each invocation of the shared
  sender. The receiver therefore receives two vectors total, one from each
  sender; both the sender and receiver use depth 2.
- Because distractor facts are removed, any improvement cannot be attributed
  solely to the smaller channel or network. The treatment jointly changes the
  input contract and model capacity as requested.

## Updates

- `2026-10-19T02:17:21Z`: Created the planned report after deterministic data
  verification and exact CPU model initialization. Launch artifacts remain
  pending immutable staging, pull-request review, and remote preflight.

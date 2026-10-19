# Tiny 32-dimensional single-link communication full run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1` |
| Status | `staged` |
| Started | not started |
| Finished | not applicable |
| Predecessor | gate retry 2 completed 200 finite steps and passed promotion checks |
| Git revision | `ceeee265c01d6431edaaf6f6125a025e12fa0918` (exact configuration commit) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml` |
| Command | pending immutable staging and final availability gate |
| Seed | `1729` |
| Hardware | selected `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; final revalidation pending |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-v1
dataset_prefix: kristine/datasets/conlangs-single-link-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
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

Relative to the passed gate, only the fresh run/output/W&B identities and
`steps` change. Duration increases from 200 to 2,000 steps. At batch 256 this
exposes 512,000 examples, approximately 5.12 passes over the 100,000-example
training split. Architecture, optimizer, data, seed, and evaluation cadence
remain fixed.

## Results

No full-run result yet. The promotion gate reduced training loss from
`13.219337` to `6.178020` and sustained median post-warmup throughput
`8,133.165` examples/s with a restorable checkpoint and clean termination.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml` | parses and differs from the gate only in four intended identity/duration fields; SHA-256 `5b514cff77fcfbc31caf81cc74020473a3ab5e9d4987cae347ae5b6656077f08` |
| Dataset manifest | `datasets/linked_facts_v1/manifest.json` | full streaming verifier passed; SHA-256 `5f95f8f4f25a10d12704959c6f275f978548b3641a4db03338d76d4f8d654a13` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1` | seven objects / 11,992,785 bytes; bucket location `US-WEST4` |
| TPU-local dataset | `/home/kristinelu/elf-data/conlangs-single-link-v1` | gate localization passed streaming verification and all seven object hashes matched |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z` at source revision `e3569ae7b232c7741a0fe4cf4fbe0ef44316b986` | 74 files / 531,437 bytes; content-manifest SHA-256 `b8615b6d4929fb345a0010c08925b516c4e0ec43695cc245b4f7481f57ab0663` |
| Transfer archive | `/tmp/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z.tar.gz` | 106,696 bytes; local and remote SHA-256 `4f7e0021c39d65870c0380607ecc4c476a168179094956ecd81626ecf12ec6b9` |
| TPU-local source | `/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z` | extracted 74 files / 531,437 bytes; no `.git`; content and config hashes match |
| Checkpoint | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/checkpoint.msgpack` | no |
| Tracking run | pending | no |
| Recurring monitor | pending first useful progress event | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The exact production model has 228,271 parameters and emits one continuous
  32-dimensional vector from each sender; sender and receiver depths are 2.
- The full run starts fresh under a new identity. It does not silently resume
  the gate checkpoint, which remains an immutable promotion artifact.
- A run-scoped recurring monitor will be installed after the first finite
  progress event because the repository has no generic watch script.

## Interpretation

- The gate established mechanical correctness and ample throughput but was too
  short for a quality conclusion. The 2,000-step run is the first meaningful
  convergence test for this tiny single-link architecture.

## Updates

- `2026-10-19T03:08:11Z`: Selected the fresh full-run identity after the gate
  passed every promotion check. The exact primary YAML was committed alone at
  `ceeee265c01d6431edaaf6f6125a025e12fa0918` and pushed to PR #2.
- `2026-10-19T03:11:10Z`: Created the immutable source snapshot, transferred it
  as a regular-file deterministic archive, matched the archive SHA-256 on the
  TPU, and matched all 74 extracted files plus the selected configuration.

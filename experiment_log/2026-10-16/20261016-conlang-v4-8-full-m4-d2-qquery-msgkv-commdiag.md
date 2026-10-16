# Full-data question-query/message-KV communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` |
| Status | `planned` |
| Started | `not recorded; not launched` |
| Finished | `not recorded; not launched` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | primary implementation/configuration commit `c7e51f00729c6bda3fb0550571778bfa17022e8a`; correctness-reviewed closure `f56d215185495e60f9bd6a69f19cb8d60e4642fb`; cleanup re-review and immutable launch snapshot pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | intended exact same authorized TPU only: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices; current availability pending |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
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

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Receiver question/message integration | Concatenate question and messages before receiver self-attention | Encode separately; question supplies queries and messages supply all keys/values | Remove question embeddings from decoder content while preserving question-directed retrieval from fixed, question-blind sender messages |

Data, seed, batch size, optimization, hidden width, output width, untied encoders, message length, depth, communication rounds, role gates, sender question blindness, answer-token objective, and causal evaluation remain fixed. No bridge or other intermediate supervision is used.

## Hypothesis and Decision Rule

The matched baseline reaches `10/256` exact answers whether messages are intact, zeroed, rotated to another world, swapped, or reduced to either sender alone. Its final sender-path gradient norms are `0.0045245` and `0.0054019`, versus `1.1364310` for the receiver. This shows that answer cross-entropy remains high while almost no useful credit reaches the question-blind senders.

The treatment keeps the senders question-blind. A shared receiver encoder separately contextualizes the question and the eight message vectors. The question representation is used only as the cross-attention query; the messages provide every key and value. The attention result has no question residual and is the only memory exposed to the answer decoder. This tests whether structurally removing question content from the decoder memory makes ordinary answer cross-entropy open the message gradient path without any additional label or auxiliary loss.

Treat the architecture as communicating only if intact messages beat both zero and wrong-world messages in answer exact match and mean joint log probability. Raw intact accuracy alone is insufficient. Compare sender-path gradients and steady-state throughput to the exact baseline, but interpret gradient magnitude only alongside the causal interventions. A same-world fixed-slot query code or global answer prior can still exist, so only intact-versus-wrong-world improvement identifies useful passage content.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml` | committed in `c7e51f00729c6bda3fb0550571778bfa17022e8a`, parses and validates locally; correctness-reviewed at `f56d215185495e60f9bd6a69f19cb8d60e4642fb`; cleanup re-review, push, and remote hash pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR; update pending |
| Source snapshot | `not recorded; pending immutable staging` | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | previously verified Central2 copy; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/checkpoint.msgpack` | no; run not launched |
| Tracking run | `not recorded; run not launched` | no |

## Review and Verification

- Focused query-routing closure passed: 4 cases, including configuration boundaries, question/value separation, nonzero message gradient, default-path preservation, and receiver-question masking.
- `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py tests/test_linked_facts_generator.py`: 67 passed with two existing Pydantic warnings in 208.27 seconds.
- Correctness passed at `f56d215185495e60f9bd6a69f19cb8d60e4642fb`; cleanup findings are fixed locally and re-review is pending.
- The first mandatory availability check on `2026-10-16` returned no final `AVAILABLE` row: the intended prior Kristine v4-8 was absent, one Kristine v6e-16 was deleting, and one Kristine v5e-4 was actively occupied and left untouched. No TPU was selected or modified.
- The filtered availability check at `2026-10-16T18:36Z` returned the same state and no final `AVAILABLE` row. The occupied v5e-4 was again left untouched.
- Immutable snapshot, remote source, runtime, regional-data, output-freshness, and VFIO checks are pending capacity.

## Failure Details

- Last known step: `not applicable; not launched`.
- Error summary: `not applicable; waiting for an authorized available Kristine TPU`.
- Detailed output: `not recorded`.

## Updates

- `2026-10-16`: Selected the question-query/message-key-value receiver as a single architectural intervention after confirming that the original receiver could remain insensitive to messages while answer cross-entropy stayed unsaturated.
- `2026-10-16`: The initial mandatory filtered audit found no available Kristine TPU. The occupied Kristine v5e-4 was left untouched; implementation and review continue while capacity is unavailable.

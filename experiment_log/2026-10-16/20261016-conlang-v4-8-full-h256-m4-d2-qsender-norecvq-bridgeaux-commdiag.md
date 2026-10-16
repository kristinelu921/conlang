# Full-data hidden-256 bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Git revision | primary configuration commit `570334c62e6ed33d0fe1f6e3b2b30061ace79b7c`; corrected reviewed configuration/report head `5e32f33d1ccee31688f915d6c2ee01163157bed8`; immutable launch snapshot pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` |
| Command | not recorded; pending immutable snapshot and launch-boundary verification |
| Seed | `1730` |
| Hardware | exact same authorized TPU only: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag
wandb_enabled: true
hidden_size: 256
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
receiver_question_input: false
communication_rounds: 1
bridge_supervision_weight: 1.0
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
| Transformer, embedding, and message-vector width | `128` | `256` | Test coupled model and continuous-channel capacity only after explicit bridge credit has produced beneficial private-passage use |

Configured output hidden width remains `256`, so the final answer vocabulary projection stays fixed; the preceding answer output-hidden input kernel widens from `(128, 256)` to `(256, 256)` with `hidden_size`. The bridge vocabulary projection also widens from `(128, vocab)` to `(256, vocab)` because it consumes the message representation directly. Message length remains four slots per sender, but each continuous message vector widens from 128 to 256 features together with the transformer and token embeddings. Data, seed, optimization, schedule, depth, untied encoders, gates, sender conditioning, receiver-question masking, communication rounds, bridge objective, hardware, and evaluation interventions remain fixed.

## Hypothesis and Decision Rule

The hidden-128 bridge-supervised arm is the first experiment in this campaign where intact messages beat target-question/wrong-passages messages: `17/256` versus `12/256`, with a `+0.0430698`-nat likelihood gap and `32/256` intact bridge accuracy. Its residual error may now reflect insufficient coupled model or continuous-channel capacity rather than the earlier missing sender credit. Because `hidden_size` controls both the transformer representation and each message vector's feature dimension, this experiment cannot isolate which of those two capacity changes causes any gain.

Doubling hidden width should improve the learned bridge representation and the receiver's two-hop join while preserving the private-passage causal gap. Treat width as useful only if intact-over-wrong-passages answer exact match remains positive and preferably exceeds `5/256`, intact bridge accuracy improves beyond `32/256`, and any raw exact-match gain is not explained by question transport alone. Reject width as the next stacked setting if the matched passage gap collapses, even if intact exact match rises. After this untied arm, compare a matched hidden-256 tied-encoder arm rather than confounding width and sharing.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` | committed in `570334c62e6ed33d0fe1f6e3b2b30061ace79b7c`, parses and validates locally, and passed correctness review after report-only corrections |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | pending push of the reviewed configuration closure |
| Source snapshot | pending | must be produced by `tpu_scripts/run_staging.sh` before launch |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | must revalidate the complete Central2 manifest before launch |
| Tracking run | pending | W&B enabled; final run identity must match the selected configuration |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack` | pending |

## Review and Verification

- The configuration parses and validates locally. Its exact YAML diff from the completed hidden-128 bridge-supervised baseline is limited to run identity/output fields plus `hidden_size: 128` to `256`.
- Correctness review passed for the committed configuration/report closure after correcting the coupled-width and output-projection descriptions.
- Cleanup review identified two report-only provenance/projection clarifications; cleanup re-review is pending this correction.
- Pending mandatory same-TPU availability check, node-local snapshot verification, regional-data integrity check, runtime preflight, and launch-boundary VFIO/process/output checks.

## Failure Details

- Last known step: not started.
- No failure; launch is pending review and preflight.

## Updates

- `2026-10-16`: Selected hidden width 256 as the next single-variable test after explicit bridge supervision established beneficial private-passage use at hidden width 128.

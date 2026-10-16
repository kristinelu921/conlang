# Full-data hidden-256 tied bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
| Baseline | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Git revision | configuration closure and immutable launch snapshot pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml` |
| Command | not recorded; pending immutable snapshot and launch-boundary verification |
| Seed | `1730` |
| Hardware | exact same authorized TPU only: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag
wandb_enabled: true
hidden_size: 256
output_hidden_size: 256
tie_sender_receiver_encoder: true
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
| Sender/receiver encoder sharing | untied | tied | Test whether a common fact/question representation improves the already active bridge-supervised communication path |

Data, seed, optimization, schedule, transformer/message width, output hidden width, message length, depth, gates, sender conditioning, receiver-question masking, communication rounds, bridge objective, hardware, and causal evaluation interventions remain fixed.

## Hypothesis and Decision Rule

The hidden-256 untied arm preserves beneficial private-passage use but does not improve over hidden width 128: intact answer accuracy is `14/256`, target-question/wrong-passages accuracy is `11/256`, the likelihood gap is `+0.0409479` nats, bridge accuracy is `34/256`, and throughput is approximately 3,249 examples/s. The remaining sharing hypothesis is that one encoder representation can align sender fact selection with receiver message interpretation and reduce redundant parameters without changing the information path.

Treat tying as useful at hidden width 256 only if the intact-over-wrong-passages answer gap remains positive and preferably exceeds the untied `3/256`, intact bridge accuracy remains above its controls and preferably exceeds `34/256`, and any raw exact-match improvement is passage-specific. Compare the tied result both to the matched hidden-256 untied arm and to the stronger hidden-128 untied bridge arm (`17/256`, `5/256` passage gap). Reject tying if it erases the passage gap or improves only question transport. Do not attribute a tied-versus-untied difference to width because width is held at 256 in both arms.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml` | parses and validates locally; immutable commit and review pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | pending update with exact configuration closure |
| Source snapshot | pending | must be produced by `tpu_scripts/run_staging.sh` before launch |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | must revalidate the complete Central2 manifest before launch |
| Tracking run | pending | W&B enabled; final run identity must match the selected configuration |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/checkpoint.msgpack` | pending |

## Review and Verification

- The configuration parses and validates locally. Its exact YAML diff from the completed hidden-256 untied bridge-supervised baseline is limited to run identity/output fields plus `tie_sender_receiver_encoder: false` to `true`.
- Pending correctness review followed by cleanup review of the immutable configuration closure.
- Pending mandatory same-TPU availability check, node-local snapshot verification, regional-data integrity check, runtime preflight, and launch-boundary VFIO/process/output checks.

## Failure Details

- Last known step: not started.
- No failure; launch is pending review and preflight.

## Updates

- `2026-10-16`: Selected encoder tying as the next single-variable test after hidden width 256 preserved private-passage use but failed the capacity stacking rule.

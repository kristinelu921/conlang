# Full-data message-4 depth-2 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Status | `planned` |
| Started | `not recorded` |
| Finished | `not recorded` |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100` |
| Git revision | `pending reviewed commit` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml` inside the immutable TPU-local snapshot; complete SSH launcher command pending staging |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
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
| Evaluation interval | 100 steps | 2000 steps | selected config |
| Causal message interventions | disabled | enabled | selected config |

Training data, batch size, optimization, seed, hidden width, output width, untied encoders, disabled role gates, four message vectors per sender, two-layer stacks, precision, and regional data are unchanged. The less frequent evaluation avoids repeated intervention overhead while preserving an exact final comparison.

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
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml` | pending reviewed commit |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `not recorded` | pending |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | previously verified; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag/checkpoint.msgpack` | pending |
| Tracking run | `not recorded` | pending |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-commdiag/metrics.jsonl` | pending |

## Failure Details

- Last known step: `not recorded`
- Error summary: not applicable; launch pending
- Detailed output: `not recorded`

## Observations

- The treatment is evaluation-only: the optimization path is byte-for-byte identical to the established full-data message-4/depth-2 baseline until final evaluation.
- `wrong_world_message` rotates messages by one verified eight-question world within each device-local validation shard. Each 64-example shard contains eight aligned worlds, so every example receives another world's messages rather than another question's identical same-world messages.
- W&B inventory at `2026-10-16T06:20:15Z` reached the entity and listed 21 projects, but every project run query was inaccessible; no W&B state was inferred from the empty result. Local immutable artifacts remain the primary evidence.

## Interpretation

- If intact messages do not outperform zero or wrong-world messages, the run has not learned example-specific communication even if its ordinary exact match moves slightly.
- Sender swapping is not expected to be harmful because each sender receives a random balanced mixture of first-hop and second-hop facts. A large swap penalty would reveal unintended positional specialization.

## Updates

- `2026-10-16T06:35:00Z`: Planned as the causal baseline after existing full-data, width-512, gated, tied/untied, and small-prefix runs all remained near a question-only answer-prior baseline.

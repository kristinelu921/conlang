# Approximately 4x sender and receiver capacity retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Supersedes | zero-step launch-control failure `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200` |
| Git revision | not recorded; pending retry closure |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml` |
| Command | not recorded; pending immutable retry snapshot |
| Seed | `1730` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; revalidation pending immediately before retry launch |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1
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

Scientific configuration is byte-for-byte identical to the failed zero-step
attempt. Only `run_id`, `output_path`, and `wandb_run_name` receive the required
fresh `retry1` identity. Relative to the completed baseline, receiver hidden,
feed-forward, output-hidden, and message widths scale `128/512/256/128` to
`256/1024/1024/256`; sender hidden/feed-forward widths scale `192/768` to
`384/1536`; attention heads scale receiver `4` to `8` and sender `6` to `12`.
Depths remain receiver 2 and sender 4. Initialized production counts remain
`7,298,176` sender, `6,986,639` receiver, and `14,845,711` total parameters.

## Hypothesis and Decision Rule

This retry preserves the original coupled-capacity hypothesis: approximately
four times as many sender and receiver parameters may let answer-token cross-
entropy discover passage-specific two-hop communication. Success requires a
repeatable intact-over-wrong-world validation advantage across the scheduled
200-step evaluations, preferably in exact match. Intact-versus-zero remains
only an out-of-distribution learned-carrier diagnostic. Because sender,
receiver, message, and output widths scale together, any gain belongs to the
coupled treatment rather than one isolated component.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | primary configuration above | parses and validates; immutable commit pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | update pending review |
| Source snapshot | not recorded; pending staging | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | prior launch preflight verified 95,042,272 bytes and manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20`; retry gate pending |
| Checkpoint | retry output path from configuration | no |
| Metrics | retry output directory | no |
| Outputs | retry output directory | no |
| Launcher log | retry run-scoped launcher log | no |
| Tracking run | pending W&B initialization | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The predecessor failed before W&B and JAX initialization because its wrapper
  created the exact scientific output directory before the trainer's freshness
  guard. No model or dataset behavior was exercised.
- This retry will direct W&B's base directory to the separate pre-existing
  `/home/kristinelu/elf-runs/wandb` path and will not create the retry output
  directory before `train.py`.

## Interpretation

- Pending results. A lower loss or larger message magnitude without a durable
  intact-versus-wrong-world gap would mean capacity still favors a non-passage-
  specific carrier or answer prior rather than communication.

## Review and Verification

- Pending immutable retry closure and the repository's correctness-then-cleanup
  review sequence.

## Updates

- `2026-10-17T20:00:48Z`: Created the fresh retry plan after the predecessor's
  zero-step launch-control failure; scientific settings are unchanged.

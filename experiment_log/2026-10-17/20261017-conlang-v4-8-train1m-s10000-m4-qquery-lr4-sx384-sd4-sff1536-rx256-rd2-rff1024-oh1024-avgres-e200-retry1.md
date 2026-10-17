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
| Git revision | `3d7597c29275bbc074cb03d26795e8bf9e9e8b13` (retry configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml` |
| Command | recorded under Exact Launch Command below |
| Seed | `1730` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; revalidation pending immediately before retry launch |

## Exact Launch Command

```bash
/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; trap "" HUP; source_dir=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1-20261017T200706.184345335Z; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; mkdir -p /home/kristinelu/elf-runs/wandb; test -w /home/kristinelu/elf-runs/wandb; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache WANDB_DIR=/home/kristinelu/elf-runs/wandb PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml 2>&1 | tee "$launcher_log"' --quiet
```

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
| Configuration | primary configuration above | parses and validates; SHA-256 `fe801cd1caf23def3d97e4da3a0bea20a92e7265e11cb9bf26cccd0b2cab3298`; committed at `3d7597c29275bbc074cb03d26795e8bf9e9e8b13` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | retry closure and both review fixes pushed through `bc4c6b9` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1-20261017T200706.184345335Z` locally; corresponding `/home/kristinelu/elf-runs/sources/` path in the command remotely | exact local/remote match: 67 files, 480,583 bytes, relative-manifest SHA-256 `2790ee0c5861bd0045f304bbace256bec8b1f8af85113bf9521940a7218f9b81`; no cache files |
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
- This retry will direct W&B's base directory to the separate sibling
  `/home/kristinelu/elf-runs/wandb` path. The exact wrapper creates and checks
  only that sibling after output freshness checks; it never creates the retry
  output directory before `train.py`.
- The first snapshot-copy invocation was rejected locally before transfer
  because TPU `scp` accepts `--scp-flag`, not `--ssh-flag`. No remote path was
  created. The corrected invocation used `--scp-flag=-F`,
  `--scp-flag=/dev/null`, and `--scp-flag=-O`; the exact-copy verification above
  then passed. This was transport-command validation, not a run attempt.

## Interpretation

- Pending results. A lower loss or larger message magnitude without a durable
  intact-versus-wrong-world gap would mean capacity still favors a non-passage-
  specific carrier or answer prior rather than communication.

## Review and Verification

- Correctness review of
  `bc1d16258fbbcb7d6171c57c18c19db19ce338d0..3d7597c29275bbc074cb03d26795e8bf9e9e8b13`
  verified the identity-only YAML delta, config validation, predecessor startup
  ordering and remote zero-step evidence, and found one launch-control issue:
  the proposed sibling W&B base did not yet exist and no exact command was
  recorded. The exact command above resolves that finding by creating only the
  sibling base after freshness checks. Cleanup review is pending.

## Updates

- `2026-10-17T20:00:48Z`: Created the fresh retry plan after the predecessor's
  zero-step launch-control failure; scientific settings are unchanged.
- `2026-10-17T20:07:06Z`: Committed and pushed the retry closure, created the
  immutable local snapshot, and corrected the wrapper to create only the
  sibling W&B base after retry-output freshness checks.
- `2026-10-17T20:09:28Z`: Copied the reviewed snapshot node-locally after one
  rejected local `scp` flag invocation, then verified exact file count, byte
  count, manifest digest, config digest, and absence of cache files remotely.

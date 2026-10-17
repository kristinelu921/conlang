# Two-hundred-step causal intervention trajectory

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag` |
| Status | `planned` |
| Started | not started |
| Finished | not started |
| Baseline | `20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Git revision | pending immutable configuration commit and launch snapshot |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml 2>&1 \| tee "$launcher_log"' --quiet` |
| Seed | `1730` |
| Hardware | selected candidate `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`; pending final launch gates |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Validation and intervention cadence | every `10,000` steps | every `200` steps | selected configuration |

Run identity, output path, and W&B display name are fresh. All scientific model,
optimizer, seed, batch, precision, data, training horizon, validation prefix,
and intervention settings remain matched to the completed baseline.

## Hypothesis and Decision Rule

The matched baseline exhibits a loss transition centered near step 830, but it
records causal message interventions only at step 10,000. This run measures the
same fixed 256-example validation prefix under intact, zero, wrong-world,
swapped-sender, agent-1-only, and agent-2-only messages every 200 training
steps. The cadence brackets the transition at steps 800 and 1,000 without
changing the training data, seed, initialization, or scientific objective.

Interpret an intact-versus-zero likelihood gap without an intact-versus-wrong-
world gap as learned nonzero-carrier dependence, not passage-specific
communication. Interpret an intact advantage over wrong-world messages as
beneficial aligned private-content use only when the sign is reproducible over
multiple evaluations and is accompanied by a meaningful exact-match or
likelihood effect. Compare single-sender and swapped-sender arms to locate
channel interference. Because evaluation itself adds wall time but performs no
optimizer update, compare training-step trajectories directly and report
throughput separately from the baseline.

## Results

Pending launch and verified completion. The expected record contains 50
evaluation points from steps 200 through 10,000, each with exact match and mean
joint log probability for all six intervention arms.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml` | local pre-launch copy; immutable commit pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing run PR; update pending |
| Source snapshot | pending staging |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | previously verified on selected Central2 TPU; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag/checkpoint.msgpack` | pending |
| Metrics | pending |
| Outputs | pending |
| Launcher log | pending |
| Tracking run | pending W&B assignment |

## Failure Details

- Last known step: not applicable; run not started.
- Error summary: not applicable.
- Detailed output: not recorded; launch pending.

## Observations

- The mandatory filtered inventory found the selected Central2 v4-8 available,
  idle, and runtime-ready before configuration authoring.
- The matched baseline completed on the same TPU and node-local dataset at a
  median final-50 throughput of `5,682.36` examples/s.

## Interpretation

- This design can establish when the receiver becomes dependent on learned
  messages and whether aligned passage content becomes causally beneficial. It
  cannot localize a change more finely than the 200-step evaluation cadence.

## Updates

- `2026-10-17`: Planned a fresh matched 10,000-step rerun with causal
  intervention validation every 200 steps.

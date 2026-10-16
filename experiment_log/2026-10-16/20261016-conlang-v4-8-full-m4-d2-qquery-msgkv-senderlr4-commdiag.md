# Question-query/message-KV sender-update balancing diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Status | `planned` |
| Started | `not recorded; not launched` |
| Finished | `not recorded; not launched` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` |
| Git revision | primary implementation/configuration commit `f60a841729595dee407d66fbaf6b6b7f1ab07ad5`; immutable reviewed launch snapshot pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag; output_dir=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag; launcher_log=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml 2>&1 \| tee "$launcher_log"'` |
| Seed | `1730` |
| Hardware | intended `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host and four TPU devices; final selection and launch gates pending |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 4.0
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag
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
| Sender-private post-Adam update multiplier | 1.0 | 4.0 | Increase actual sender parameter movement without changing the answer-token objective, receiver optimizer, shared token embedding, model, data, or seed |

The implementation also logs sender, receiver, embedding, and total update
norms plus sender/receiver update-to-weight ratios. This telemetry does not
change optimization, but the historical baseline did not record these fields
and did not pay their logged-step computation cost.

## Hypothesis and Decision Rule

The matched query/message-KV run starts from sender and receiver weight norms of
`60.0652142` and `107.1733465`, with per-parameter RMS values `0.0952922` and
`0.0816988`. The receiver has `4.33x` as many private parameters, but that size
difference is far too small to explain the final approximately `330x` raw path-gradient
ratio. By step 2000, sender weight norm changes only `+0.04%`, while receiver
weight norm rises approximately `+8.0%`. This argues against a simple initial
weight-scale mismatch and instead suggests persistent end-to-end credit
attenuation along the longer sender path.

Adam normalizes each coordinate by its gradient moments, so multiplying raw
sender gradients by a constant would mostly cancel inside Adam. This treatment
therefore multiplies the sender-private update after AdamW by `4.0`, including
its proportionate weight-decay update, while leaving receiver and shared-token
updates unchanged. It adds no labels or auxiliary objective.

Treat balancing as a communication improvement only if intact messages beat
both zero and wrong-world messages in exact match and mean joint log
probability. Interpret sender/receiver update-to-weight ratios within this run;
the historical baseline cannot supply a matched update-ratio comparison.
Compare weight-norm drift, path-gradient norms, loss, and causal evaluation to
the exact query/message-KV baseline. Report throughput descriptively, but do
not attribute its difference solely to the multiplier because the new logged-
step update telemetry adds instrumentation work. A larger sender update without
an intact-versus-wrong-world advantage is optimizer activity, not evidence of
useful content transport.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` | parses and validates locally; immutable reviewed source pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | update pending |
| Source snapshot | `not recorded; pending immutable staging` | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | prior Central2 copy verified; fresh preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack` | no; run not launched |
| Tracking run | `not recorded; run not launched` | no |

## Review and Verification

- Focused optimizer, default-compatibility, telemetry, and validation closure: 5 passed with two existing Pydantic warnings in 28.14 seconds.
- `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py tests/test_linked_facts_generator.py`: 70 passed with the same two existing warnings in 203.53 seconds.
- Required sequential correctness and cleanup reviews are pending the immutable implementation/configuration commit.
- The mandatory filtered TPU audit completed with three Kristine v4-8 candidates and one Kristine v5e-8 candidate in its final `AVAILABLE` table. Runtime probes returned `UNKNOWN`; the prior Central2 v4-8 remains only an intended candidate until exact readiness and occupancy gates pass.

## Failure Details

- Last known step: `not applicable; not launched`.
- Error summary: `not applicable`.
- Detailed output: `not recorded`.

## Updates

- `2026-10-16`: Existing query/message-KV dynamics show comparable initialization scales but persistent sender credit attenuation and essentially flat sender weight norm. Selected a `4x` sender-private post-Adam update treatment with direct update-norm telemetry.

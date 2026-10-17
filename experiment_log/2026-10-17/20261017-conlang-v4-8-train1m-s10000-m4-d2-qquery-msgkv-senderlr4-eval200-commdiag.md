# Two-hundred-step causal intervention trajectory

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag` |
| Status | `running` |
| Started | `2026-10-17T03:02:11Z` |
| Finished | not started |
| Baseline | `20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Git revision | primary configuration/report commit `6003960c6e9c3caffe5d1b2ef044d35e9d604180`; reviewed immutable launch snapshot `967a9be97c9b4ddd7a55d967901f07c5a81d523c` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml 2>&1 \| tee "$launcher_log"' --quiet` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`; READY, HEALTHY, idle, and 84 GiB free before launch |

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
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag.yaml` | staged SHA-256 `7e0f6a93fc5ef3898e396ca69d8ba3605f7221be51f92b8e9f35bfcfc4a1e17e` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | branch updated through launch closure `967a9be97c9b4ddd7a55d967901f07c5a81d523c` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag-20261017T025836.329941057Z`; remote `/home/kristinelu/elf-runs/source/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag` | local and remote match: 63 files, 462255 bytes, relative SHA-256-manifest digest `489054a3ea9e475cc70bed19af058a8a0034d9b41992e08b3768e7852b8988b8`; no cache files |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | freshly verified: 128125 worlds, 1025000 questions, 8 questions/world; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag/checkpoint.msgpack` | pending |
| Metrics | pending | no |
| Outputs | pending | no |
| Launcher log | pending | no |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/dr3ydegs` | run ID `dr3ydegs`; live sync confirmed at launch |

## Failure Details

- Last known step: not applicable; run not started.
- Error summary: not applicable.
- Detailed output: not recorded; launch pending.

## Observations

- The mandatory filtered inventory found the selected Central2 v4-8 available,
  idle, and runtime-ready before configuration authoring.
- Final preflight found the TPU READY/HEALTHY and idle with 84 GiB free. The
  runtime has JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, and W&B `0.28.1`.
- The matched baseline completed on the same TPU and node-local dataset at a
  median final-50 throughput of `5,682.36` examples/s.

## Interpretation

- This design can establish when the receiver becomes dependent on learned
  messages and whether aligned passage content becomes causally beneficial. It
  cannot localize a change more finely than the 200-step evaluation cadence.

## Review and Verification

- The selected YAML parses and validates through `TrainConfig`; it differs
  from the completed matched baseline only in fresh identity/output fields and
  `eval_every: 10000` to `200`.
- Focused CPU verification passed three intervention tests in `38.09s` with
  `USE_TORCH_XLA=0` and `JAX_PLATFORMS=cpu`.
- Correctness review found no confirmed issue in
  `466a54096d90c5095b3872deb231f77d6eac3b33..95c6686cabbee89cf0159abbb492f2c71cc109d3`.
  Cleanup review found two report-table issues; both are fixed in
  `16e97a980009eb858922187454c60bbd6ae70625`.

## Updates

- `2026-10-17`: Planned a fresh matched 10,000-step rerun with causal
  intervention validation every 200 steps.
- `2026-10-17T02:58:36Z`: Staged immutable source commit `967a9be`, copied it
  once to node-local TPU storage, and verified source, configuration, runtime,
  regional data, disk, and idle-device launch gates.
- `2026-10-17T03:02:11Z`: Launched on the selected v4-8. W&B run `dr3ydegs`
  began syncing, and the exact trainer process was confirmed active on worker 0.

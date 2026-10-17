# Capacity-matched shared sender with an average residual — retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Status | `preflight passed; launch pending` |
| Started | not started |
| Finished | not applicable |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-eval200-commdiag` |
| Retry of | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200` |
| Git revision | `53151f17b14841f6275793a96459785ec59720d9` (reviewed configuration closure used for the immutable snapshot) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1-20261017T153818.926176475Z; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.yaml 2>&1 | tee "$launcher_log"' --quiet` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; API state READY/HEALTHY, one host/four TPU devices, 85 GiB free, runtime and occupancy gates passed |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
sender_hidden_size: 192
sender_num_layers: 4
sender_num_heads: 6
sender_feedforward_size: 768
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
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Sender hidden width | `128` | `192` | selected configuration |
| Sender encoder depth | `2` | `4` | selected configuration |
| Sender attention heads | `4` | `6` | selected configuration |
| Sender feed-forward width | `512` | `768` | selected configuration |
| Sender average residual | disabled | enabled | selected configuration |
| Sender parameters | `397,312` | `1,830,464` | initialized production Flax parameter tree |

This retry changes only run identity, output path, and W&B display name from the
retired zero-step attempt. Receiver dimensions, four-by-128 message channel,
optimizer, sender learning-rate multiplier, seed, batch, precision, regional
dataset, 10,000-step horizon, fixed validation prefix, 200-step intervention
cadence, and causal controls remain matched. Both passage agents reuse one
sender parameter tree. The treatment intentionally couples sender capacity and
the average-residual initialization.

## Hypothesis and Decision Rule

The matched baseline learns a sharp sender-gradient pulse near step 830 but no
stable passage-specific communication. The selected shared sender has four
192-wide layers, six 32-dimensional heads, and a TPU-aligned 768-wide FFN. It
projects back to the unchanged 128-dimensional message channel. Four masked
contiguous passage-region averages provide the initial messages exactly; the
learned sender projection begins at zero and can acquire a residual correction
without an auxiliary target or sender-side question input.

The treatment succeeds only if intact messages acquire a reproducible advantage
over wrong-world messages across multiple scheduled evaluations, preferably
with an exact-match improvement. Intact-versus-zero alone remains an
out-of-distribution carrier-dependence diagnostic. Compare the step-830 region,
sender/receiver gradient and relative-update ratios, message variation,
single-sender arms, output collapse, and final validation against the baseline.
Because width/depth and the average residual change together, attribute a
positive result to the combined treatment rather than either component.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.yaml` | local and remote SHA-256 `4a0fb81af1d6a1b4845537305b8ba7415c405c0063dd475467603123e4eaf7f8`; parsed and validated remotely |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR; update pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1-20261017T153818.926176475Z`; remote `/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1-20261017T153818.926176475Z` | local and remote match: 65 files, 477,532 bytes, relative manifest digest `c8b9145d6e91773b4b3b228ae12610c9e3ddd23ee282a61b180ccc64c07a04ea`; no bytecode/test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | generated on the selected Central2 TPU; 128,125 worlds / 1,025,000 questions / 8 questions per world; 95,042,272 bytes; all six gzip streams and content checks passed; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/checkpoint.msgpack` | no |
| Metrics | pending run-scoped artifact directory | no |
| Outputs | pending run-scoped artifact directory | no |
| Launcher log | pending run-scoped artifact directory | no |
| Tracking run | not recorded | no |

## Failure Details

- Last known step: not applicable.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- Retry 0 failed only at the local initializer credential-path preflight and
  made no cloud call or remote change. This retry passes the separately
  verified owner-only `/home/kristinelu/.netrc` via `ELF_NETRC_PATH`; the
  credential remains excluded from the source snapshot.
- The required filtered inventory selected an idle region-matched Central2
  v4-8 from its final `AVAILABLE` table. Direct describe reported READY,
  HEALTHY, Spot `v4-8`; the read-only occupancy helper reported AVAILABLE.
- The baseline receiver has `1,720,847` parameters. The selected sender has
  `1,830,464` parameters, 6.37% more than the unchanged receiver. The full
  initialized model has `3,831,759` FP32 parameters (`15,327,036` raw bytes).
- The regional bucket reports location `US-CENTRAL2`, but
  `gs://kmh-gcp-us-central2/kristine/datasets/conlangs-train1m` contains no
  objects. Matching the previous 1M run, the deterministic dataset was
  regenerated on the selected Central2 host with seed `1729`, starter split
  sizes except `train_worlds=125000`, then fully stream-verified. No dataset
  crossed regions.
- The installed runtime reports JAX/JAXlib `0.6.2`, Flax `0.10.5`, Optax
  `0.2.8`, NumPy `1.26.4`, and W&B `0.28.1`; the runtime freeze digest is
  `775959a9a56fe178a814793e9bf32ad6c779933d1f4900cbb2239a8e18cd1346`.
  The repository preflight and a separate CPU-only integrity probe confirmed
  output/log freshness, source/config hashes, W&B authentication, writable
  local paths, no canonical libtpu lock, and idle VFIO resources.

## Interpretation

- Pending results. A lower loss without a durable intact-versus-wrong-world gap
  would mean greater sender capacity or the average bypass improves optimization
  without establishing useful private-content transport.

## Review and Verification

- The selected YAML parses and validates through `TrainConfig`. Relative to the
  completed eval-200 baseline, its only scientific changes are the five listed
  sender fields; relative to retry 0, only identity fields differ.
- Production initialization verified `3,831,759` total parameters with exactly
  one `1,830,464`-parameter sender tree shared across both passage calls.
- Focused CPU verification passed 9 sender, initialization, sharing, shape, and
  causal-intervention tests in `46.91s`.
- The full CPU suite passed `142` tests and `7` subtests with two existing
  Pydantic warnings in `204.71s` under `USE_TORCH_XLA=0` and
  `JAX_PLATFORMS=cpu`.
- Correctness review found no confirmed issue in immutable range
  `930cc28689ad3d4a2d3b24fb487b570a2a140db4..e236107a0e0cd814e9b32c06c2ad9bfdded29701`.
  Cleanup review found one unused set of sender override fields on
  `EncoderStack`; commit `dbc5ed661c1647501e4dd24e1dad0c8f4e2f5744`
  removed them. Seven focused tests passed in `30.12s`, then the full
  `142`-test/`7`-subtest suite passed again in `216.43s` with the same two
  existing warnings.

## Updates

- `2026-10-17T15:27Z`: Created retry 1 after the isolated-worktree credential
  preflight stopped retry 0 before cloud access. Scientific settings are
  unchanged; run, output, and W&B identities are fresh.
- `2026-10-17T15:38:18Z`: Created and copied the immutable 65-file source
  snapshot, then verified identical local and remote manifests and selected
  config hashes.
- `2026-10-17T15:42Z`: Finished deterministic node-local dataset generation
  and full integrity verification. The isolated runtime and final prelaunch
  integrity gates passed; launch remains pending the immediate final occupancy
  recheck.

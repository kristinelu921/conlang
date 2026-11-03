# Capacity-matched shared sender with an average residual — retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Status | `completed` |
| Started | `2026-10-17T15:44:16Z` |
| Finished | `2026-10-17T16:11:10Z` |
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

Training completed all `10,000` optimizer steps. The run emitted `1,000`
training records at ten-step intervals and `50` evaluations of the same fixed
256-example validation prefix at 200-step intervals.

| Final training metric | Value |
|---|---:|
| Loss | `4.0355186` |
| Total gradient norm | `0.7791419` |
| Agent 1 / agent 2 path-gradient norm | `0.1207619` / `0.1497786` |
| Receiver path-gradient norm | `0.7649000` |
| Sender / receiver relative update norm | `0.00157285` / `0.000246879` |
| Agent 1 / agent 2 message RMS | `4.8659577` / `4.8848920` |
| Throughput | `2,376.52` examples/s |

| Final validation intervention | Exact match | Count | Mean joint log probability |
|---|---:|---:|---:|
| Intact messages | `0.01953125` | 5/256 | `-5.9475145` |
| Wrong-world messages | `0.01953125` | 5/256 | `-5.9499369` |
| Swapped senders | `0.01953125` | 5/256 | `-5.9427977` |
| Zero messages | `0.00000000` | 0/256 | `-11.9979849` |
| Agent 1 only | `0.00000000` | 0/256 | `-5.5615034` |
| Agent 2 only | `0.00000000` | 0/256 | `-5.5275278` |

The run fails the preregistered communication rule. Intact and wrong-world
messages have identical exact-match counts in every one of the 50 evaluations.
The intact-minus-wrong-world likelihood gap is positive in 28/50 evaluations,
but its median is only `+0.0000348` nats, with range `-0.0014191` to
`+0.0024223`. The final `+0.0024223`-nat aligned-passage advantage is therefore
small, sign-unstable over training, and not accompanied by any exact-answer
advantage. The `+6.0504704`-nat intact-versus-zero gap establishes dependence
on a learned nonzero carrier, not passage-specific content. Swapping channels
slightly improves final likelihood by `0.0047169` nats, while either isolated
sender improves likelihood but gets no exact answer.

The capacity/residual treatment removes raw sender-gradient starvation without
repairing passage use. Over the last 50 logged steps, median agent path-gradient
norms are `0.1356820` and `0.1500125`, versus `0.7444219` for the receiver:
receiver ratios of only `5.49x` and `4.96x`. The matched baseline ratios are
`13,005.91x` and `13,558.64x`. Median sender relative update is `0.00179503`,
`10.67x` the receiver's `0.000168306`; the baseline sender is only `0.623x` its
receiver. Treatment sender relative updates are `17.27x` baseline, while
receiver relative updates are essentially unchanged (`1.009x`). Message RMS
is `3.83x` baseline. From step 10 to 10,000, sender and receiver weight norms
grow `66.07%` and `78.01%`; baseline sender norm instead falls `4.27%`.

The largest non-initial adjacent five-record-versus-five-record loss decline
after step 100 begins at step 980 (`5.2955249` before versus `4.9101499` after),
about 150 steps later than the matched baseline's step-830 transition. A second
smaller decline appears near step 3330 (`4.1111673` to `3.9435777`). Neither is
a communication transition: validation likelihood worsens from `-5.3896303` at
step 200 to `-5.9020844` at step 1,000 and `-11.3220463` at step 3,400, while
the wrong-world exact count remains tied. Late optimization is also less
stable: the maximum logged pre-clipping gradient norm is `132.1732` at step
8,920, and 11 logged steps exceed 10, versus a baseline maximum of `25.3769`
and two such records. All values remain finite and training completes.

Median final-50 loss is `3.9889567`, effectively the same as baseline
`3.9882981`, while median throughput falls from `5,681.41` to `2,380.07`
examples/s (`-58.11%`). The final logged training batch emits `walrus` for
255/256 examples and `jaguar` once. This output artifact records training-batch
generations; validation generations are not persisted.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1.yaml` | local and remote SHA-256 `4a0fb81af1d6a1b4845537305b8ba7415c405c0063dd475467603123e4eaf7f8`; parsed and validated remotely |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | PR branch contains completed-evidence commit `663c6d5844f61829d8de9d291e3fab2440f6618d` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1-20261017T153818.926176475Z`; remote `/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1-20261017T153818.926176475Z` | local and remote match: 65 files, 477,532 bytes, relative manifest digest `c8b9145d6e91773b4b3b228ae12610c9e3ddd23ee282a61b180ccc64c07a04ea`; no bytecode/test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | generated on the selected Central2 TPU; 128,125 worlds / 1,025,000 questions / 8 questions per world; 95,042,272 bytes; all six gzip streams and content checks passed; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/checkpoint.msgpack` | 46,001,025 bytes; SHA-256 `01c7b39f7498184f79b28f43573ba01237c1194911b7e7a215f4e74863daff87`; CPU restore verified step `10000` |
| Metrics | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/metrics.jsonl` | 1,984,543 bytes; SHA-256 `068a92268c87d7020aab35f93a4e64198633c3a837b9b1c4beb6066863cdb501`; 1,050 JSON records parsed |
| Outputs | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/outputs.jsonl` | 13,825,638 bytes; SHA-256 `439600f2baff8a2236f89d056e31fb38a063c0b5e9d4b8c63e6ef89a49f4c046`; 1,000 JSON records parsed |
| Launcher log | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1/launcher.log.gz` | 14,755 bytes; SHA-256 `e539b9222653b1b4ef42f849662e719e7db2ca1d4afd0e9ebd6d8f7db611349e`; gzip verified; decompressed 85,558-byte SHA-256 `77d2bc5d0e9e70f3d24fd8ed80261ba53666740447b85299a9d772ba66f4e92e` matches remote canonical log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/hqwvdzrr` | run ID `hqwvdzrr`; API state `finished`, step `10000`, final metrics match, five files synced |

## Failure Details

- Last known step: `10000` (successful completion).
- Error summary: no training failure.
- Detailed output: the canonical launcher log contains the completion sentinel
  and no traceback.

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
- The first artifact-copy attempt over the external SSH endpoint packet-failed
  after metrics completed. The canonical remote artifacts were never modified.
  The launcher log was recovered through the internal endpoint; the large
  outputs file was transferred as a remote gzip, locally decompressed, and
  accepted only after its decompressed byte count and SHA-256 matched the
  original. The temporary transfer gzip is not part of the committed evidence.
- One monitor iteration used an invalid `gcloud describe` formatter and was
  replaced by a corrected read-only monitor. Training was not signaled or
  restarted. Subsequent checks and the terminal audit reported READY/HEALTHY,
  and the final audit found no trainer, no lock, and idle VFIO resources.

## Interpretation

- The combined capacity/average-residual treatment decisively fixes the raw
  sender-credit imbalance and drives much larger messages, updates, and weight
  growth. It does not create useful private-content transport: wrong-world is
  exactly tied at every scheduled evaluation, and its likelihood effect remains
  tiny and sign-unstable. The failure is therefore not explained by sender
  gradient magnitude or aggregate sender movement alone.
- Because sender depth, width, FFN size, and residual initialization change
  together, this run cannot attribute its optimization dynamics to one of those
  components. It does establish that simply making question-blind shared
  senders receiver-sized, initializing a strong average carrier, and training
  for 10,000 steps is insufficient under answer-token cross-entropy alone.
- The treatment is not a useful default: it is `58.11%` slower, produces larger
  late gradient excursions, collapses the final logged training batch to two
  outputs, and leaves final-50 loss unchanged. The five final exact answers do
  not show communication because wrong-world preserves all five.

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
- A separate completed-result correctness review found no confirmed issue in
  exact range
  `3194e4b7059bf89b079c0fd7cd70900228c3370b..663c6d5844f61829d8de9d291e3fab2440f6618d`.
  It independently recomputed all metrics and comparisons, parsed both JSONL
  artifacts, verified artifact/checkpoint/source/config/dataset/runtime hashes,
  restored checkpoint step 10,000, confirmed W&B completion, checked causal
  language and secrets, and found a clean diff and worktree. The completed-result
  cleanup pass found only stale PR-publication and review-ledger wording; this
  follow-up records the already pushed evidence commit and the distinct result
  review.

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
- `2026-10-17T15:44:16Z`: Launched the exact recorded command on
  `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`; W&B run
  `hqwvdzrr` initialized. Data loading confirmed 1,000,000 training examples,
  256 validation examples, batch 256, bfloat16, and four TPU devices. The run
  reached step 50 with loss `5.4793`, gradient norm `1.5060`, and approximately
  `2,413` examples/s.
- `2026-10-17T16:11:10Z`: Training completed step 10,000 with a saved and
  restorable checkpoint, finalized metrics and outputs, completion sentinel,
  and W&B state `finished`. The recurring monitor stopped only after the
  terminal process, checkpoint, log, W&B, VFIO, lock, TPU state, and artifact
  checks passed.

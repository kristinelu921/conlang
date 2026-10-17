# Approximately 4x sender and receiver capacity

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200` |
| Status | `failed` |
| Started | `2026-10-17T20:00:48Z` |
| Finished | `2026-10-17T20:00:48Z` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Git revision | `4cca1d6a61ed409e476ad880f290badbde1ec775` (launch snapshot; configuration/report closure at `97d0c9256d639b21bed05b978f2040eec4df99fa`) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml` |
| Command | recorded under Exact Launch Command below |
| Seed | `1730` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; current API state READY/HEALTHY and occupancy AVAILABLE |

## Exact Launch Command

```bash
/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; trap "" HUP; source_dir=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-20261017T195535.297491701Z; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; mkdir -p "$output_dir/wandb"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache WANDB_DIR="$output_dir/wandb" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml 2>&1 | tee "$launcher_log"' --quiet
```

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Receiver hidden width | `128` | `256` | selected configuration |
| Receiver attention heads | `4` | `8` | selected configuration |
| Receiver feed-forward width | `512` | `1024` | derived as `4 * hidden_size` |
| Output-hidden width | `256` | `1024` | selected configuration |
| Sender hidden width | `192` | `384` | selected configuration |
| Sender attention heads | `6` | `12` | selected configuration |
| Sender feed-forward width | `768` | `1536` | selected configuration |
| Message vector width | `128` | `256` | coupled to receiver hidden width |
| Sender parameters | `1,830,464` | `7,298,176` (`3.987x`) | initialized production Flax trees |
| Receiver parameters | `1,720,847` | `6,986,639` (`4.060x`) | initialized production Flax trees |
| Total parameters | `3,831,759` | `14,845,711` (`3.874x`) | initialized production Flax trees |

Run identity, output path, and W&B display name are fresh. All other scientific
and operational fields remain matched: sender depth 4,
receiver depth 2, four message slots per sender, average-passage residual,
question-blind shared sender, receiver question-only query path, sender learning
rate multiplier 4, optimizer, seed, 1M-example Central2 dataset, global batch
256, bfloat16 precision, 10,000 steps, fixed 256-example validation prefix,
and 200-step causal evaluation cadence.

## Hypothesis and Decision Rule

The previous treatment removed raw sender-gradient starvation but did not learn
passage-specific communication. This arm tests whether approximately four times
as many parameters in each sender and receiver, plus the naturally wider
message carrier and output head, lets answer-token cross-entropy discover the
two-hop communication circuit without auxiliary labels.

The treatment succeeds only if intact messages develop a repeatable advantage
over wrong-world messages across multiple scheduled evaluations, preferably in
exact match. Intact-versus-zero remains only an out-of-distribution learned-
carrier diagnostic. Compare the intact/wrong-world gap trajectory, final-50
sender/receiver gradients and relative updates, message RMS, output diversity,
validation stability, and throughput with the exact baseline. Because receiver,
sender, message, and output widths change together, any improvement belongs to
the coupled scaling treatment and cannot identify one component as causal.

## Results

The wrapper exited before W&B initialization, JAX distributed initialization,
dataset loading, model initialization, or step 1. No scientific result was
produced. The TPU remained unclaimed and is eligible for the corrected retry.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.yaml` | parsed, validated, and CPU-initialized; SHA-256 `2554d0b48d5c74a943884459069e5781adfdbced25faaa7d5357358d8290e640`; committed at `97d0c9256d639b21bed05b978f2040eec4df99fa` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR updated through `4cca1d6a61ed409e476ad880f290badbde1ec775` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-20261017T195535.297491701Z` locally; `/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-20261017T195535.297491701Z` remotely | exact match: 66 files, 479,047 bytes, relative-manifest SHA-256 `cb8f09f63c9f5123b1488e855254b0e68d2ca202ea3b5914eceff008d8774508`; no cache files |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | 95,042,272 bytes; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20`; all gzip streams and generator verifier pass: 128,125 worlds, 1,025,000 questions, 8 questions/world |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200/checkpoint.msgpack` | absent |
| Metrics | run-scoped output directory | absent |
| Outputs | run-scoped output directory | absent |
| Launcher log | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200.launcher.log` | 998 bytes; preserved with traceback |
| Tracking run | not created | trainer freshness guard ran before W&B initialization |

## Failure Details

- Attempt / retry: original attempt; corrected run is `retry1`.
- TPU / zone / topology: selected Central2 Spot v4-8 above.
- Stage reached and last step: launch wrapper; zero training steps.
- First error and exit status: `FileExistsError` at `train.py:1044`; launcher
  exited 1 because the run output directory already existed.
- Detailed log path: launcher log recorded in Artifacts.
- Confirmed root cause and evidence: the launch wrapper created
  `$output_dir/wandb` before calling `train.py`; the trainer intentionally
  rejects any pre-existing output directory. The only output entry is the
  empty `wandb/` directory, and the log contains the freshness traceback.
- Alternative causes rejected: no trainer survived, no libtpu lock exists,
  every VFIO device is idle, and no checkpoint, metric, output, config copy, or
  W&B run was created.
- Packages before / after: unchanged.
- Repair command: use a fresh `retry1` identity and place W&B files under the
  pre-existing `/home/kristinelu/elf-runs/wandb` parent without creating the
  scientific output directory before `train.py`.
- Verification command and result: exact process, lock, VFIO, output-tree, and
  launcher-log audit passed; only the audit shell itself matched the run ID.
- Checkpoint / artifact state: no checkpoint; failed output and launcher log
  preserved in place.
- Recovery action: prepare and immutably stage a fresh retry configuration.
- Prevention gate added: wrapper review now checks that no preparation command
  creates `config.output_path.parent` before the trainer freshness guard.

## Observations

- CPU production initialization with vocabulary size `2191` gives `14,845,711`
  FP32 parameters: `560,896` shared embedding, `7,298,176` shared sender, and
  `6,986,639` receiver.
- Sender and receiver attention retain 32-dimensional heads. All selected
  hidden, feed-forward, output, and per-head dimensions are multiples of 32;
  feed-forward and output widths are multiples of 128.
- The required filtered inventory returned five AVAILABLE Kristine TPUs. The
  selected Central2 v4-8 already holds the verified regional dataset and runtime
  from the matched baseline. A direct describe reports READY/HEALTHY, and the
  read-only occupancy helper reports AVAILABLE.
- Required remote preflight passed with 85 GiB free, no libtpu lock, all VFIO
  devices idle, a fresh output/log path, JAX `0.6.2`, Flax `0.10.5`, Optax
  `0.2.8`, NumPy `1.26.4`, W&B `0.28.1`, and a valid W&B netrc entry.
- The first read-only dataset-content verifier was malformed by controller-side
  shell quoting and exited with Python `SyntaxError` before reading content. It
  created no trainer, W&B run, output, checkpoint, lock, or TPU holder. The
  corrected module verifier then passed the counts recorded above; this was a
  prelaunch verification-command incident, not a training attempt.

## Interpretation

- This attempt produced no scientific result, so its launch-control failure has
  no bearing on the capacity hypothesis. The unchanged hypothesis and decision
  rule carry forward to the fresh `retry1` run.

## Review and Verification

- Preliminary CPU validation: the YAML parses through `TrainConfig`, all head
  divisibility and node-local path constraints pass, and the production model
  initializes with the stated parameter counts and a 4-by-256 message from each
  sender.
- Correctness review found no confirmed issue in immutable range
  `708eee6b258df8e9f9049087ad0490b88d8693ae..97d0c9256d639b21bed05b978f2040eec4df99fa`.
  It independently verified the YAML delta, production shapes and parameter
  counts, finite forward and greedy paths, seven focused tests, static
  per-device batch, HBM plausibility, causal rule, scope, and secrets. Cleanup
  review found only stale commit-state wording and an overbroad matched-fields
  sentence; this follow-up records the immutable commit and explicitly excludes
  the three fresh identity/path fields.

## Updates

- `2026-10-17T19:43:50Z`: Selected the approximately 4x-capacity treatment,
  confirmed current filtered Kristine TPU availability, and created the planned
  report. No TPU state was changed and no training process was started.
- `2026-10-17T19:58:57Z`: Staged and byte-verified the immutable source on the
  selected TPU, freshly verified the regional dataset, and passed the official
  and run-specific remote preflight gates. Training remained unstarted.
- `2026-10-17T20:00:48Z`: Launch exited before training because the wrapper
  pre-created the output directory for W&B. Preserved the zero-step attempt,
  proved the TPU and runtime remained idle, and began a fresh `retry1` closure.

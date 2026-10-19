# Tiny 32-dimensional single-link one-epoch 2.56M run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1` |
| Status | `completed` |
| Started | `2026-10-19T03:58:33Z` |
| Finished | `2026-10-19T04:10:08Z` (canonical launcher-log modification time after W&B synchronization) |
| Baseline | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1` completed 2,000 steps on the 100,000-example dataset |
| Git revision | source snapshot from `1e2808fe3dcee9cc7cda039f023713164e4d6883`; exact configuration commit `74c6486328e2f3e6a0514943da126ff2e9f63989` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml` |
| Command | exact foreground command in Launch Command below |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; READY/HEALTHY at launch |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml`

```yaml
run_id: 20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-n2560000-v1
dataset_prefix: kristine/datasets/conlangs-single-link-n2560000-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 10000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1
wandb_enabled: true
hidden_size: 32
output_hidden_size: null
sender_hidden_size: 32
sender_num_layers: 2
sender_num_heads: 4
sender_feedforward_size: 128
sender_average_residual: false
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: false
receiver_question_input: true
receiver_question_query_only: false
communication_rounds: 1
bridge_supervision_weight: 0.0
message_length: 1
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
| Training examples | 100,000 | 2,560,000 | verified dataset manifests |
| Steps | 2,000 | 10,000 | primary configurations |
| Epochs | 5.12 effective passes | 1 exact pass | steps × batch size / training examples |
| Batch size | 256 | 256 | primary configurations |
| Message vectors per sender | 1 | 1 | primary configurations |
| Continuous message width | 32 | 32 | primary configurations |
| Sender / receiver depth | 2 / 2 | 2 / 2 | primary configurations |

All optimizer hyperparameters and seed remain unchanged.

## Dataset Generation

Exact command:

```bash
python -m linked_facts.generate_dataset \
  --output-dir /tmp/conlangs-single-link-n2560000-v1 \
  --seed 1729 \
  --entities-per-world 8 \
  --train-worlds 320000 \
  --validation-worlds 625 \
  --test-worlds-per-split 625 \
  --compression gzip
```

Full verification command:

```bash
python -m linked_facts.generate_dataset \
  --output-dir /tmp/conlangs-single-link-n2560000-v1 \
  --verify-only
```

Verification result: 323,125 worlds, 2,585,000 total questions, and 8 questions per world. The training split contains 320,000 worlds and exactly 2,560,000 questions. Each sender input contains one necessary fact; the two sender facts together form the two-hop chain.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `manifest.json` | `d8576253b65786a99f5143582c8b561b063a3ecb239ff6b54044286f1854835f` | 3,087 |
| `train.jsonl.gz` | `f267ed081a4ead1886a353967d19b2204800bbb60a7b8b05d392e5c812e3a247` | 244,578,956 |
| `validation.jsonl.gz` | `79c51d6784694af833f4227121e424829d2aa0e233e318a354b0ec9e5619e049` | 480,142 |
| `test_familiar.jsonl.gz` | `10a6df68c5a649efb9ee29f11494daa426c403e77d61d71dd2fedcc72897cfe3` | 479,873 |
| `test_new_names.jsonl.gz` | `a7ce0a2b2ba53747c94a00e79ed765b05affa151cdcecf2185bdc65b52bdcb70` | 485,327 |
| `test_new_wording.jsonl.gz` | `f40080851ab8e5baaae23148164eaf1f2c594ac9674754f59c889a0452b7284e` | 491,162 |
| `test_new_names_and_wording.jsonl.gz` | `e4b428586fb3ff04bcac10ab3276abdd5b2d73769397c7f957b62096cb319adf` | 499,394 |

Regional source: `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-n2560000-v1` in `us-west4`. The upload inventory contains exactly seven objects and 247,017,941 bytes, matching the local file sizes.

## Validation

- Configuration parse and arithmetic assertions passed: 10,000 steps, batch size 256, and 2,560,000 training presentations.
- Architecture assertions passed: message length 1, continuous width 32, sender depth 2, and receiver depth 2.
- Full dataset streaming validation passed, including world structure, counts, byte sizes, and uncompressed content checksums.
- Focused configuration/batching tests: 17 passed.
- Broader generator and JAX training suite: `77 passed` with `JAX_PLATFORMS=cpu` in 216.74 seconds.
- TPU-local dataset verification passed with the same 323,125 worlds, 2,585,000 total questions, 8 questions per world, and all seven local SHA-256 hashes matching the generation host.

## Review

The required correctness pass reviewed the complete PR diff from base
`051db79bc7092a3c5422f3ace9e7aaa42c32d53e` through head
`c300d249ea614affd6478f595626e2582183a7d8` and returned
`NO_CONFIRMED_FINDINGS`. Its focused generator, model-shape, and staging checks
passed 12 tests. The cleanup pass reviewed updated head
`59766e18101c1374572adb0735c7518af58da57c`, found two report-maintenance
issues, and both were applied before staging. No third review was performed.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | loss | 2.312741 | 10,000 / 1.0 | min | `metrics.jsonl` |
| training | examples / second | 7,882.12 | 10,000 / 1.0 | max | terminal steady-state interval in `metrics.jsonl` |
| validation, 256 examples | exact match | 0.589844 | 10,000 / 1.0 | max | `metrics.jsonl` and W&B summary |
| validation, 256 examples | mean joint log probability | -2.303925 | 10,000 / 1.0 | max | `metrics.jsonl` |
| validation, agent 1 message only | exact match | 0.308594 | 10,000 / 1.0 | max | `metrics.jsonl` |
| validation, agent 2 message only | exact match | 0.332031 | 10,000 / 1.0 | max | `metrics.jsonl` |
| validation, swapped sender messages | exact match | 0.589844 | 10,000 / 1.0 | max | `metrics.jsonl` |
| validation, wrong-world messages | exact match | 0.031250 | 10,000 / 1.0 | max | `metrics.jsonl` |
| validation, zero messages | exact match | 0.042969 | 10,000 / 1.0 | max | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Source configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml` | parses; SHA-256 `a5bc2c80fcabfa7a2564cc4ff061fdb9caaaea76c7207eaff4c40e403aa32b66` |
| Effective configuration | `artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/config.yml` | 1,538 bytes; SHA-256 `87ee03be04c0f3926b6a8cc2614d78096512a85096a5cc9f532ef8b72a58e2d9` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-n2560000-v1` | seven objects / 247,017,941 bytes; full manifest and local hashes verified |
| TPU-local dataset | `/home/kristinelu/elf-data/conlangs-single-link-n2560000-v1` | all seven SHA-256 hashes and full streaming verifier passed |
| Immutable source | `/home/kristinelu/elf-runs/sources/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-20261019T035122.898087842Z` | 75 files; content hash `ea9f31967eecf44def18bda18f10a614c3c06378b9829427d68006f54a1e8d72` |
| Transfer archive | `/home/kristinelu/elf-runs/source-archives/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-20261019T035122.898087842Z.tar.gz` | 106,990 bytes; SHA-256 `ed8c783f853d3c8819e41a68a71df4c32925319279dbdc2541004f88cc606973` |
| Checkpoint | `/home/kristinelu/elf-runs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/checkpoint.msgpack`; local workspace copy at `/kmh-nfs-ssd-us-mount/code/kristine/conlang/experiment_log/2026-10-19/artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/checkpoint.msgpack` | restored payload step `10000`; 2,751,863 bytes; SHA-256 `673c8c83985b390e374af55bd9e4d1a9b71bb670d3dc701bc9b59741825ccf8a` |
| Metrics | `artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/metrics.jsonl` | 1,050 records; 2,753,715 bytes; SHA-256 `57759a503d830a762038e31ac8b1287978d8b4b4bc3c8b6a51999db1375e0095` |
| Sampled model outputs | `artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/outputs.jsonl` | 1,000 records; 13,993,661 bytes; SHA-256 `28147586c7ac1d033dc8de6431c43e83e07128de78cb2bdb393ec533e8b50393` |
| Canonical launcher log | remote raw log plus `artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/launcher.log.gz` | completion sentinel occurs once; raw log is 84,311 bytes with SHA-256 `ac8fce169a8acf0d60b600134c9516502fd11774a2e5731cf4a65ee2858bb12d`; deterministic gzip is 16,314 bytes with SHA-256 `feab8c7fe31f05ee41c46aba0dbdab30ce5414ada7170c85a8a1ffbe47c41be8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/cebrgj40` | `finished`; summary step `10000`; summary validation exact match `0.58984375` |
| Monitor | `artifacts/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/monitor.sh` and `monitor.log` | checks passed while active; recurring timer stopped and verified `inactive` after terminal verification |
| Plots / report | this report | yes |

## Failure Details

- Last known step: 10,000, with a restored checkpoint payload at the same step.
- Error summary: not applicable; the run completed successfully.
- Detailed output: `/home/kristinelu/elf-runs/logs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.launcher.log`.

## Observations

- The training split has exactly 2,560,000 verified examples, equal to 10,000
  full batches of 256 with no partial batch.
- Each sender input contains exactly one necessary fact and produces one
  continuous 32-dimensional message vector.
- The selected TPU-local dataset matches the generation host byte-for-byte.
- Corrected TPU-VM CPU-only initialization with the production tokenizer
  confirmed message shape `(2, 2, 32)`, 228,271 parameters, sender depth 2,
  and receiver depth 2.
- The trainer reported all 8 TPU devices, exactly 2,560,000 training examples,
  and finite metrics through terminal step 10,000.
- After completion, the exact trainer process was absent, all eight VFIO
  resources and the libtpu lock were idle, and the TPU remained READY/HEALTHY.
- The canonical launcher log contains exactly one matching
  `TRAINING_COMPLETE` sentinel. W&B independently reports the run as finished
  at summary step 10,000.

## Interpretation

- The 10,000-step run completed exactly one data epoch because 10,000 batches
  multiplied by 256 examples per batch equals all 2,560,000 examples.
- Terminal validation exact match is 58.98%. Agent-1-only and agent-2-only
  exact match are 30.86% and 33.20%, while wrong-world and zero-message
  controls fall to 3.12% and 4.30%, respectively.
- Swapping the two sender messages leaves exact match unchanged at 58.98% on
  this fixed validation sample, consistent with order-insensitive use of the
  two one-fact messages for this task.

## Updates

- `2026-10-19T03:49:35Z`: Generated, uploaded, localized, and fully verified
  the 2.56M-example training split; local CPU regression suite passed 77 tests.
- `2026-10-19T03:49:35Z`: Correctness review returned no confirmed findings.
  Cleanup review found report-schema issues; both findings were applied before
  immutable staging. No third review was performed, per the two-pass policy.
- `2026-10-19T03:55:33Z`: Preserved a failed optional pytest-based preflight in
  `20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-preflight-attempt1.md`.
  The production runtime intentionally lacks pytest; the direct replacement
  model assertion passed without changing the runtime or creating run output.
- `2026-10-19T04:05:08Z`: Foreground training reached finite step 300 at
  7,807.42 examples/second. W&B run `cebrgj40` is active, and the independent
  five-minute monitor completed its first check and armed the next trigger.
- `2026-10-19T04:10:08Z`: Training emitted its unique completion sentinel at
  step 10,000. The checkpoint restored at step 10,000, W&B finalized as
  `finished`, TPU state remained READY/HEALTHY, and every VFIO resource and the
  libtpu lock was idle. Final artifacts were copied and checksum-verified, then
  the recurring monitor was stopped and verified inactive.

## Launch Command

```bash
gcloud compute tpus tpu-vm ssh \
  kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a \
  --project=he-vision-group \
  --zone=us-west4-a \
  --worker=all \
  --command='set -euo pipefail; trap "" HUP; source_path=/home/kristinelu/elf-runs/sources/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-20261019T035122.898087842Z; config_path="$source_path/configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml"; launcher_log=/home/kristinelu/elf-runs/logs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.launcher.log; mkdir -p /home/kristinelu/elf-runs/logs /home/kristinelu/elf-runs/wandb /home/kristinelu/elf-data/huggingface; cd "$source_path"; export JAX_PLATFORMS=tpu; export CUDA_VISIBLE_DEVICES=; export ELF_TPU_WORKER_INDEX=0; export PYTHONPATH="$PWD"; export HF_HOME=/home/kristinelu/elf-data/huggingface; export WANDB_DIR=/home/kristinelu/elf-runs/wandb; export PYTHONUNBUFFERED=1; export PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config "$config_path" 2>&1 | tee "$launcher_log"' \
  --quiet
```

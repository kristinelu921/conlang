# Tiny 32-dimensional single-link communication full run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1` |
| Status | `completed` |
| Started | `2026-10-19T03:13:55Z` |
| Finished | `2026-10-19T03:18:11Z` |
| Predecessor | gate retry 2 completed 200 finite steps and passed promotion checks |
| Git revision | `ceeee265c01d6431edaaf6f6125a025e12fa0918` (exact configuration commit) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml` |
| Command | exact command below |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; terminal state READY/HEALTHY and idle |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-v1
dataset_prefix: kristine/datasets/conlangs-single-link-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
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

## Canonical Effective Configuration

TPU-written `output/config.yml`, copied verbatim:

```yaml
batch_size: 256
bridge_supervision_weight: 0.0
communication_rounds: 1
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-v1
dataset_dir: null
dataset_prefix: kristine/datasets/conlangs-single-link-v1
device: tpu
dropout: 0.1
eval_every: 200
eval_examples: 256
evaluation_message_interventions: true
hidden_size: 32
learning_rate: 0.0003
limit_examples: null
log_every: 10
max_answer_length: 4
max_gradient_norm: 1.0
max_question_length: 128
max_text_length: 512
message_length: 1
num_heads: 4
num_layers: 2
output_hidden_size: null
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/checkpoint.msgpack
precision: bfloat16
pretrained_checkpoint_sha256: null
pretrained_initialization: null
pretrained_model_dir: null
pretrained_tokenizer_sha256: null
receiver_question_input: true
receiver_question_query_only: false
relative_attention_max_distance: 128
relative_attention_num_buckets: 32
repeat_training_examples_to_batch_size: false
role_gate_channels: null
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
seed: 1729
sender_average_residual: false
sender_feedforward_size: 128
sender_hidden_size: 32
sender_learning_rate_multiplier: 1.0
sender_num_heads: 4
sender_num_layers: 2
sender_question_conditioning: false
steps: 2000
throughput_warmup_steps: 10
tie_sender_receiver_encoder: false
wandb_enabled: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1
weight_decay: 0.01
```

## Launch Command

```bash
gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a \
  --project=he-vision-group \
  --zone=us-west4-a \
  --worker=all \
  --command='set -euo pipefail; trap "" HUP; source_path=/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z; config_path="$source_path/configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml"; launcher_log=/home/kristinelu/elf-runs/logs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1.launcher.log; mkdir -p /home/kristinelu/elf-runs/logs /home/kristinelu/elf-runs/wandb /home/kristinelu/elf-data/huggingface; cd "$source_path"; export JAX_PLATFORMS=tpu; export CUDA_VISIBLE_DEVICES=; export ELF_TPU_WORKER_INDEX=0; export PYTHONPATH="$PWD"; export HF_HOME=/home/kristinelu/elf-data/huggingface; export WANDB_DIR=/home/kristinelu/elf-runs/wandb; export PYTHONUNBUFFERED=1; export PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config "$config_path" 2>&1 | tee "$launcher_log"' \
  --quiet
```

## Changed Hyperparameters

Relative to the passed gate, only the fresh run/output/W&B identities and
`steps` change. Duration increases from 200 to 2,000 steps. At batch 256 this
exposes 512,000 examples, approximately 5.12 passes over the 100,000-example
training split. Architecture, optimizer, data, seed, and evaluation cadence
remain fixed.

## Results

The full run completed all 2,000 finite steps. Training loss fell from
`13.219337` at step 10 to `3.478937` at step 2,000. Across 199 post-warmup
measurements, median throughput was `8,319.639` examples/s; the minimum was
`6,758.739` examples/s at an evaluation boundary. The maximum logged
pre-clipping gradient norm was `9.834501`.

Validation exact match rose monotonically in broad trend across the scheduled
checks: `0.0000`, `0.0117`, `0.0234`, `0.0586`, `0.0781`, `0.1562`, `0.1562`,
`0.1875`, `0.1914`, and `0.2812` at steps 200 through 2,000. Final mean joint
log probability was `-3.366223`.

At step 2,000, full communication scored `0.28125` exact match. Agent-1-only
and agent-2-only were each `0.183594`; swapping sender messages scored
`0.292969`; wrong-world messages scored `0.035156`; zero messages scored
`0.027344`. The model therefore uses information from both messages and is
strongly sensitive to their content, while the sender ordering itself is not
important in this configuration.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-full1.yaml` | parses and differs from the gate only in four intended identity/duration fields; SHA-256 `5b514cff77fcfbc31caf81cc74020473a3ab5e9d4987cae347ae5b6656077f08` |
| Dataset manifest | `datasets/linked_facts_v1/manifest.json` | full streaming verifier passed; SHA-256 `5f95f8f4f25a10d12704959c6f275f978548b3641a4db03338d76d4f8d654a13` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1` | seven objects / 11,992,785 bytes; bucket location `US-WEST4` |
| TPU-local dataset | `/home/kristinelu/elf-data/conlangs-single-link-v1` | gate localization passed streaming verification and all seven object hashes matched |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z` at source revision `e3569ae7b232c7741a0fe4cf4fbe0ef44316b986` | 74 files / 531,437 bytes; content-manifest SHA-256 `b8615b6d4929fb345a0010c08925b516c4e0ec43695cc245b4f7481f57ab0663` |
| Transfer archive | `/tmp/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z.tar.gz` | 106,696 bytes; local and remote SHA-256 `4f7e0021c39d65870c0380607ecc4c476a168179094956ecd81626ecf12ec6b9` |
| TPU-local source | `/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1-20261019T031013.568489761Z` | extracted 74 files / 531,437 bytes; no `.git`; content and config hashes match |
| Effective configuration | `/kmh-nfs-ssd-us-mount/code/kristine/conlang/experiment_log/2026-10-18/artifacts/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/config.yml` | 1,501 bytes; SHA-256 `1ca9d676236a9f33e6f632a27432c1c8786da753d53ed503faf6dd2361d5ec8c`; copied verbatim below |
| Checkpoint | shared-mount artifact directory above and TPU-local `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1/checkpoint.msgpack` | 2,751,863 bytes; SHA-256 `a7d760f01d2e983cfef0587b2cda12d372c905a6561659c0de01f58b7bab5bdf`; CPU restore verified step 2,000 |
| Metrics | shared-mount `metrics.jsonl` | 448,799 bytes; SHA-256 `618136e93ade46453eef0c354c5a9da24c507a8066484c030a2e25dd9f6beb46`; 210 records: 200 training and 10 evaluation |
| Outputs | shared-mount `outputs.jsonl` | 2,749,662 bytes; SHA-256 `b0c6b79c032978548f0094d4bdef028f1befdb146a905107a48191f16c53af8b`; 200 records; local-only because it contains dataset-derived text |
| Launcher log | shared-mount `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1.launcher.log` | 19,475 bytes; SHA-256 `5295b800f2fd18e42f5b2e1009ad080340c733334629aa9a370fd1d139354b97`; completion sentinel present |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/m41r83dd` | API state `finished`, summary step 2,000, exact match `0.28125`, five files synced |
| Recurring monitor | preserved shared-mount `monitor.sh` and `monitor.log` | clean-environment checks observed the exact trainer at step 1,470 and terminal no-trainer/checkpoint state; log 1,624 bytes, SHA-256 `feee53970862ba90e76e8e0952526c32b376fabd3385837244aa04dbfc024388`; timer retired after independent verification |

## Failure Details

- Last known step: 2,000, completed.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The exact production model has 228,271 parameters and emits one continuous
  32-dimensional vector from each sender; sender and receiver depths are 2.
- The full run starts fresh under a new identity. It does not silently resume
  the gate checkpoint, which remains an immutable promotion artifact.
- A run-scoped recurring monitor was installed after the first finite progress
  event because the repository has no generic watch script.
- The installed monitor pins the exact TPU, cloud project, zone, run identity,
  gcloud executable, launcher log, and checkpoint path. Its first verified
  check found trainer PID `219859` as the sole holder of every VFIO device and
  the libtpu lock, with finite progress through step 1,470.
- Its terminal check then found no trainer, the step-2,000 completion sentinel,
  and the 2,751,863-byte checkpoint. Independent checks restored that
  checkpoint on CPU, found no VFIO or lockfile holder, confirmed W&B
  `finished`, and found the TPU READY/HEALTHY.
- The checkpoint, effective config, metrics, generated outputs, launcher log,
  monitor script, and monitor evidence were copied to the unique shared-mount
  artifact directory and rehashed against the TPU-local originals.

## Interpretation

- The 32-dimensional, one-vector-per-sender model learns the task measurably:
  exact match reaches `28.125%` after 5.12 training passes and loss continues
  downward. This is substantially above both wrong-world and zero-message
  controls, so the receiver is using communicated content rather than solving
  from the question alone.
- Either single sender retains useful partial information (`18.359%` each),
  while both together add about 9.77 percentage points. Swapping the two
  messages does not hurt, which is consistent with the two sender channels
  being treated symmetrically rather than encoding a fragile fixed role order.
- This run establishes learning but not convergence: the final validation
  curve is still improving. A longer follow-up could test the attainable
  ceiling without changing the architecture, but no additional run is implied
  by this completed request.

## Updates

- `2026-10-19T03:08:11Z`: Selected the fresh full-run identity after the gate
  passed every promotion check. The exact primary YAML was committed alone at
  `ceeee265c01d6431edaaf6f6125a025e12fa0918` and pushed to PR #2.
- `2026-10-19T03:11:10Z`: Created the immutable source snapshot, transferred it
  as a regular-file deterministic archive, matched the archive SHA-256 on the
  TPU, and matched all 74 extracted files plus the selected configuration.
- `2026-10-19T03:13:55Z`: The required final audit listed the selected v5e in
  its `AVAILABLE` table. A final direct gate confirmed fresh output/log paths,
  exact config and dataset hashes, and no process or accelerator holder; the
  foreground trainer then initialized W&B run `m41r83dd`.
- `2026-10-19T03:17:55Z`: Activated and validated the five-minute recurring
  monitor. Its clean-environment check found READY/HEALTHY cloud state, the
  exact trainer as the only accelerator holder, finite loss, and step 1,470.
- `2026-10-19T03:18:11Z`: Completed 2,000 finite steps and all ten scheduled
  intervention evaluations, saved the final checkpoint, emitted the completion
  sentinel, and finished W&B synchronization.
- `2026-10-19T03:19:48Z`: The monitor and independent terminal checks found no
  trainer or accelerator holder, restored the checkpoint at step 2,000,
  confirmed W&B `finished` and TPU READY/HEALTHY, copied and rehashed all key
  artifacts on the shared mount, and retired the run-scoped monitor.

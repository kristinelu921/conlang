# Tiny 32-dimensional single-link communication gate retry 2

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2` |
| Status | `completed` |
| Started | `2026-10-19T03:00:11Z` |
| Finished | `2026-10-19T03:06:03Z` |
| Predecessor | retry 1 failed during source transfer before copying a file |
| Git revision | `d2c3bc74df162b6545937935b8cf5c5ae24ea4d7` (exact configuration commit) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml` |
| Command | exact command below |
| Seed | `1729` |
| Hardware | selected `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; READY/HEALTHY and idle at retry selection |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml`

```yaml
run_id: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-v1
dataset_prefix: kristine/datasets/conlangs-single-link-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 200
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2
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

## Launch Command

```bash
gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a \
  --project=he-vision-group \
  --zone=us-west4-a \
  --worker=all \
  --command='set -euo pipefail; trap "" HUP; source_path=/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2-20261019T024709.880205475Z; config_path="$source_path/configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml"; launcher_log=/home/kristinelu/elf-runs/logs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.launcher.log; mkdir -p /home/kristinelu/elf-runs/logs /home/kristinelu/elf-runs/wandb /home/kristinelu/elf-data/huggingface; cd "$source_path"; export JAX_PLATFORMS=tpu; export CUDA_VISIBLE_DEVICES=; export ELF_TPU_WORKER_INDEX=0; export PYTHONPATH="$PWD"; export HF_HOME=/home/kristinelu/elf-data/huggingface; export WANDB_DIR=/home/kristinelu/elf-runs/wandb; export PYTHONUNBUFFERED=1; export PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config "$config_path" 2>&1 | tee "$launcher_log"' \
  --quiet
```

## Changed Hyperparameters

No scientific hyperparameter changed from retry 1. Only run, output, W&B, and
immutable staging identities change. Retry 2 transfers a SHA-256-recorded
regular-file archive rather than passing a `/.` recursive source to SCP.

Relative to operational throughput baseline
`20261014-conlang-jax-v5e8-b256-opt1`, both sender and receiver shrink from
width 128 to 32, each sender emits one vector instead of four, feed-forward
width shrinks from 512 to 128, and each sender sees one required fact instead
of eight mixed facts. Depth remains 2 on both sides.

## Gate Decision Rule

The gate must initialize on all TPU devices, complete 200 finite steps, save a
restorable checkpoint, and produce the scheduled validation interventions.
Promotion to a fresh full-run identity additionally requires no unexplained
material steady-state throughput regression.

## Results

The gate completed all 200 finite steps. Training loss fell from `13.219337`
at step 10 to `6.178020` at step 200. The 19 post-warmup measurements had
median throughput `8,133.165` examples/s and minimum throughput `8,087.281`
examples/s. The maximum logged pre-clipping gradient norm was `9.834501`.

Scheduled evaluation at step 200 produced exact match `0.0` and mean joint
log probability `-6.049064`. All message interventions also had zero exact
match at this short gate. This is not a convergence result; the gate passes
because the model initialized, trained with finite improving loss at high
throughput, ran every scheduled intervention, saved a restorable checkpoint,
and terminated cleanly.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.yaml` | parses and validates; SHA-256 `85973ed1985e439aa878f657b5c30af3d3bbe468b1c5be35176892c772f3c1f3` |
| Dataset manifest | `datasets/linked_facts_v1/manifest.json` | full streaming verifier passed; SHA-256 `5f95f8f4f25a10d12704959c6f275f978548b3641a4db03338d76d4f8d654a13` |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1` | seven objects / 11,992,785 bytes; bucket location `US-WEST4` |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2-20261019T024709.880205475Z` at source revision `d8859b3144c09cc77115d31c4c7aa4defee211b9` | 73 files / 529,936 bytes; content-manifest SHA-256 `f223e88e5569acddf763788e9317ceaadede790d5145549ca2a5fd640ea8a0b2` |
| Transfer archive | `/tmp/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2-20261019T024709.880205475Z.tar.gz` | 104,913 bytes; local and remote SHA-256 `919c10e0e8b57cbfb9e2a3682d845ff886aa9a081637a9b980a79f6763269ec8` |
| TPU-local source | `/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2-20261019T024709.880205475Z` | extracted 73 files; no `.git`; content and config hashes match |
| TPU-local dataset | `/home/kristinelu/elf-data/conlangs-single-link-v1` | streaming verification passed; 7 files / 11,996,881 filesystem bytes; all seven object hashes match the regional source |
| Checkpoint | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/checkpoint.msgpack` | 2,751,862 bytes; SHA-256 `4daedb62609842289f033e8ed0a1f7888f2df8a9af31813da090c2669d3c8c4a`; CPU restore verified step 200 |
| Metrics | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/metrics.jsonl` | 36,744 bytes; SHA-256 `5a4fd8a008a91d47c245a376bc40566989c7b8a43514485ee63ec6abdbe9650d`; 20 training records and 1 evaluation record |
| Outputs | `/home/kristinelu/elf-runs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2/outputs.jsonl` | 234,075 bytes; SHA-256 `9c3ef371f81bc1a9bfd98290810e6880f0d3a6c44190e5099484b11d98fd87ed` |
| Launcher log | `/home/kristinelu/elf-runs/logs/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry2.launcher.log` | 4,952 bytes; SHA-256 `fb39bfb741368a42decd7ca80624928334a8257cf948fcab8eb6aa8a35b9895d`; canonical completion sentinel present |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/aazp4xrd` | API state `finished`, summary step 200, exact match `0.0`, five files synced |

## Failure Details

- Last known step: 200, completed.
- Error summary: not applicable to retry 2; both predecessor staging failures
  have distinct immutable reports.
- Detailed output: not recorded.

## Observations

- The selected candidate remained READY, HEALTHY, Spot, and VFIO idle.
- Remote runtime imports passed with JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`,
  and W&B `0.28.1`.
- Preflight reported 4 GiB node-local free space, safely above the 12 MB
  dataset, sub-megabyte source, and small-model checkpoint/log budget.
- The exact model has 228,271 parameters and emits message shape
  `(batch, 2, 32)`: one vector from each sender.
- The TPU VM's CPU-only production-runtime preflight independently confirmed
  the retry configuration, message shape, layer depths, parameter count, and
  presence of W&B authentication without initializing the TPU backend.
- The repository has no `tpu_scripts/watch_elf_run.sh`. This six-minute gate
  completed under its attached foreground session while the fallback recurring
  monitor path was being prepared; direct process, VFIO, cloud-health,
  checkpoint, log, and W&B checks covered terminal verification.
- Terminal verification found no trainer or accelerator holder. The TPU
  remained READY/HEALTHY and the W&B API independently reported `finished`.

## Interpretation

- Retry 2 is scientifically identical to its predecessors. The fresh identity
  preserves both zero-step staging failures while using a transfer method with
  an explicit archive hash and extracted file-manifest verification.
- The gate meets its promotion rule. The throughput margin is large relative
  to the prior width-128 operational baseline, and there is no unexplained
  regression or instability. A fresh 2,000-step full-run identity is the next
  selected action; the gate checkpoint is retained as an immutable artifact
  rather than reused as an implicit resume.

## Updates

- `2026-10-19T02:46:31Z`: Selected retry 2 after preserving retry 1's empty
  remote target and repeated deterministic SCP error; exact config parsed.
- `2026-10-19T02:51:45Z`: Copied the deterministic tar archive as a regular
  file, matched its SHA-256 remotely, extracted it into the unique node-local
  target, and matched all 73 source files plus the selected configuration.
- `2026-10-19T02:51:45Z`: Localized the seven-object dataset from the
  region-matched `US-WEST4` bucket. The full stream verifier reported 15,625
  worlds and 125,000 questions, and every local object SHA-256 matched.
- `2026-10-19T02:51:45Z`: CPU-only runtime preflight confirmed two 32-wide
  sender vectors, two sender layers, two receiver layers, 228,271 parameters,
  batch 256, 200 gate steps, and available W&B authentication.
- `2026-10-19T03:00:11Z`: The required final audit listed the selected v5e in
  its `AVAILABLE` table. A final direct check found no VFIO, lockfile, or
  `train.py` holder and confirmed fresh output and launcher-log paths.
- `2026-10-19T03:00:11Z`: Started the foreground trainer from the exact
  TPU-local snapshot. W&B initialized run `aazp4xrd` under the configured run
  name; backend compilation and the first training step are pending.
- `2026-10-19T03:06:03Z`: Completed 200 finite steps and scheduled message
  interventions, saved the checkpoint, emitted the exact completion sentinel,
  and finished W&B synchronization.
- `2026-10-19T03:08:11Z`: Restored the checkpoint on CPU at step 200, recorded
  hashes and sizes for all artifacts, confirmed W&B state `finished`, found no
  trainer or accelerator holder, and confirmed the TPU remained READY/HEALTHY.

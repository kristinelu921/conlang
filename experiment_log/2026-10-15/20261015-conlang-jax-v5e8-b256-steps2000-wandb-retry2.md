# Conlang JAX v5e-8 batch-256 2,000-step W&B retry 2

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2` |
| Status | `failed` |
| Started | not applicable; blocked at the final occupancy gate |
| Finished | `2026-10-15T02:44:27Z` |
| Baseline | `20261015-conlang-jax-v5e8-b256-steps2000`; retry of zero-step `20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1` |
| Git revision | config commit `6460194eeeccbed50f173740c9e0bac5cf1a4ab4`; immutable launch tree `bfad80a1505c45cf51f575e5aafb67d0dc857182` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a --project=he-vision-group --zone=us-west4-a --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2-20261015T024203.549932560Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-steps2000-wandb-retry2.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, topology `2x4`, expected 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry2.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 100
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2
wandb_enabled: true
hidden_size: 128
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

No scientific or telemetry changes from retry 1. Only the run ID, output path, W&B display name, and immutable snapshot identity are fresh.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | baseline median steady-state throughput | 5581.683917 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `20261015-conlang-jax-v5e8-b256-steps2000` |
| training | sustained-throughput floor | 6463.955546 examples/s | post-warmup intervals | higher is better | retained 80% floor from pre-telemetry full run |
| training | loss | not recorded | not recorded | lower is better | pending |
| validation | exact match | not recorded | not recorded | higher is better | pending |
| validation | mean joint log probability | not recorded | not recorded | higher is better | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry2.yaml` | yes; parsed and validated, SHA-256 `f1a0b3b861d564a22dff2ae8b638239b39e82bc4a7e714a8c8a25ccf432f344d`, committed alone as `6460194` |
| Effective configuration | none | no; the interpreter never started |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; retry config commit pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2-20261015T024203.549932560Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 34 files, 316627 bytes, relative SHA-256-manifest digest `50ec6c3dd636dabde589e04a05f33f0a45e36c4535c7643c4fc47344197e1c0f` matched locally and remotely |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, sourced from `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; bucket location `US-WEST4`; 7 files, 11643131 bytes, 15625 worlds and 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2/checkpoint.msgpack` | no; the output directory is absent |
| Tracking run | none | no; W&B initialization was never reached |
| Metrics | none | no; the output directory is absent |
| Evaluation outputs | none | no; the output directory is absent |
| Canonical log | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry2.launcher.log` | no; the foreground fan-out was never opened |

## Failure Details

- Last known step: `0`; step 1 did not start
- Error summary: the canonical final occupancy helper returned `OCCUPIED` after a current-user LaVin-DiT validation job acquired every VFIO device between staging and launch
- Detailed output: read-only process audit identified parent PID `2272907` and eight multiprocessing children `2273030`, `2273033`–`2273039` for `lavin_dit_bench_v2_native_p4_v5e8_opt`, validation attempt v9

## Observations

- The fresh mandatory inventory selected the runtime-ready us-west4 Kristine v5e-8 from the final `AVAILABLE` table.
- Preflight found 4 GiB free on the data and output filesystems, an absent fresh output path, and idle VFIO.
- The snapshot is byte-identical to retry 1 except for the added retry-2 YAML; `train.py` SHA-256 remains `b5051fecefbec1fecb413e39e510c266ceed695d22191e44165f4523aa9b104c`.
- The canonical occupancy helper remains the process-identity authority. The redundant self-matching `pgrep` predicate from retry 1 is absent; output, lock, VFIO, and `/tmp/tpu_logs` checks remain in the launch wrapper.
- The final gate found a healthy unrelated current-user LaVin-DiT validation workload, owned by PID `2272907`, with eight child processes holding every VFIO device. It was left untouched. Retry 2 has no output directory, launcher log, W&B run, checkpoint, or training process.

## Interpretation

- This attempt produced no scientific or telemetry result. The fail-closed occupancy gate prevented a collision with an active healthy workload.
- Recovery requires waiting for authorized capacity, rerunning the mandatory filtered inventory, and using a fresh identity after the selected TPU is genuinely available.

## Updates

- 2026-10-15T02:42:03Z: Created a fresh retry identity after preserving the zero-step wrapper failure. The exact config was committed and pushed, the mandatory inventory again selected the available runtime-ready v5e-8, and the new immutable source matched locally and remotely.
- 2026-10-15T02:44:27Z: The final occupancy helper found the selected TPU newly occupied by an unrelated healthy current-user LaVin-DiT validation. The process owner, parent/child PIDs, full command, validation identity, and VFIO holdings were recorded read-only; the workload was left untouched and no retry-2 trainer or artifact was created.

## Terminal Reconciliation

- This identity was blocked before the launch wrapper or trainer started and is
  not a runnable entry point in the finalized change.
- Its YAML was removed from the finalized configuration closure. The historical
  source remains available in Git commit `6460194`.
- The pending result fields above are terminally resolved as not applicable
  because the run never launched.
- Successor evidence is recorded in
  `20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5.md`.

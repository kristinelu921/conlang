# Conlang JAX v4-8 batch-256 2,000-step W&B retry 4

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4` |
| Status | `failed before step 1` |
| Baseline | `20261015-conlang-jax-v5e8-b256-steps2000`; capacity fallback after three zero-step/unlaunched W&B attempts |
| Git revision | config commit `9b2bcf6b`; immutable launch tree `1916dcc8f4eadc202078eb009f6f298db9335450` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=0 --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4-20261015T034141.246967082Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-wandb-retry4.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-wandb-retry4.yaml`

The complete configuration is preserved in the committed source file above. Scientific and telemetry settings match retry 3; only the run identity, output path, W&B name, snapshot, and TPU generation/region change.

## Changed Hyperparameters

No model, optimizer, seed, batch, step-count, evaluation, or telemetry hyperparameter changed. W&B remains enabled. Compact 32-bin activation histograms are logged every 10 steps; vocabulary-sized token histograms and their large table are omitted.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | loss | pending | pending | lower is better | pending |
| training | throughput | pending | pending | higher is better | pending |
| validation | exact match | pending | pending | higher is better | pending |
| validation | mean joint log probability | pending | pending | higher is better | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-wandb-retry4.yaml` | yes; SHA-256 `676afdfa810a9ac90e69531d91d3a003f8c73185c8bc0a4776de5d5152038ff7`, committed alone as `9b2bcf6` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; config commit pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4-20261015T034141.246967082Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 36 files, 318903 bytes, relative SHA-256-manifest digest `5d687c4a3f85adb32b8a2c45c8371d579bafe0bbd4f3a89c9a29637ce102c843` matched locally and remotely; `train.py` SHA-256 `b5051fecefbec1fecb413e39e510c266ceed695d22191e44165f4523aa9b104c` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, generated deterministically on the central2 TPU because the matching regional bucket prefix is absent | yes; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` matched the established dataset; 7 files, 11643131 bytes, 15625 worlds, 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4/checkpoint.msgpack` | no; only `config.yml` exists in the output directory |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/pcutw300` | yes; W&B ID `pcutw300`, finalized failed before step 1 |
| Metrics | none | no metric step was produced |
| Evaluation outputs | none | no model output was produced |
| Canonical log | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4.launcher.log` | yes; 2772 bytes, SHA-256 `74fb2c1b39e91a399d95d705176dfe52d5ac75ac4237fbccf3a530354c1bc309` |

## Observations

- The required filtered inventory listed this idle Kristine v4-8 in its final `AVAILABLE` table.
- The preferred west4 v5e-8 stayed occupied by a healthy LaVin-DiT validation for more than 40 minutes; the same-region v5e-4 was also occupied. Both workloads were identified read-only and left untouched.
- The central2 `kristine/datasets/conlangs` bucket prefix is absent. The deterministic dataset will be regenerated node-locally from seed 1729 and must match the known manifest before training.
- The isolated runtime installed JAX 0.6.2, Flax 0.10.5, Optax 0.2.8, NumPy 1.26.4, and W&B 0.28.1 with authenticated W&B access.
- The JAX compatibility probe reported TPU platform, four local/global devices, and one process. Batch 256 is divisible by the device count. Preflight reported 85 GiB free for data and outputs, an absent fresh output path, and idle accelerators.
- W&B initialized successfully at `2026-10-15T03:45:29Z`, proving authenticated tracking on the new runtime. Dataset resolution then failed before loading examples or starting step 1 because Cloud SDK 384.0.1 has no `gcloud storage` command.
- The exact diagnostic returned `Invalid choice: 'storage'`; the same image's `gsutil 5.10` successfully reported the bucket location as `US-CENTRAL2`. The terminal audit found no checkpoint, metrics, evaluation output, TPU holder, or libtpu lockfile.

## Failure Details

- Last known step: `0`; step 1 did not start
- Error: `subprocess.CalledProcessError` from `gcloud storage buckets describe gs://kmh-gcp-us-central2 --format=value(location)` with exit status 2
- Root cause: the legacy v4 Cloud SDK predates the `gcloud storage` surface, while the regional-data resolver assumed that newer command existed
- Recovery: add a narrowly gated `gsutil` fallback, test it, preserve this failed W&B identity, and relaunch under fresh retry-5 output and W&B identities

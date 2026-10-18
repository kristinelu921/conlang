# Conlang JAX v4-8 batch-256 2,000-step W&B retry 5

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5` |
| Status | `completed successfully` |
| Started | `2026-10-15T03:57:01Z` |
| Finished | `2026-10-15T04:01:50Z` |
| Baseline | `20261015-conlang-jax-v5e8-b256-steps2000`; fresh retry after retry 4 failed before step 1 on legacy Cloud SDK compatibility |
| Git revision | compatibility fix `39af628`; config commit `64c77f5`; immutable launch tree `5ead6774437766c996c7cde2fdd2982bbb120355` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=0 --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5-20261015T035459.278342060Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-wandb-retry5.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-wandb-retry5.yaml`

The complete configuration is preserved in the committed source file above. Scientific and telemetry settings match retry 4; only the run identity, output path, W&B name, and immutable snapshot change.

## Changed Hyperparameters

No model, optimizer, seed, batch, step-count, evaluation, or telemetry hyperparameter changed. W&B remains enabled. Compact 32-bin activation histograms are logged every 10 steps; vocabulary-sized token histograms and their large table are omitted.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | final loss | 3.934626579 | 2000 | lower is better | `metrics.jsonl` and W&B |
| training | median steady-state throughput | 5822.019561 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| training | mean steady-state throughput | 5806.751187 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| validation | exact match | 0.0390625 | 2000 | higher is better | `metrics.jsonl` and W&B |
| validation | mean joint log probability | -4.053215981 | 2000 | higher is better | `metrics.jsonl` and W&B |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-wandb-retry5.yaml` | yes; SHA-256 `7fd6e0a335020d609c4926fc127adebff6c25370ee8e2e440dd1d31e2adc3c5d`, committed alone as `64c77f5` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; compatibility fix and retry config pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5-20261015T035459.278342060Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 37 files, 323495 bytes, relative SHA-256-manifest digest `8a23cb9b99a4c349a0d69817c896deb8890b06bdbe989ad6e200dba5a1e2d9c6` matched locally and remotely; `train.py` SHA-256 `b5051fecefbec1fecb413e39e510c266ceed695d22191e44165f4523aa9b104c`; `regional_data.py` SHA-256 `2f350cd45f653c60defd00dece9017b1edf6180e2e357744fe89857c25ca8f1d` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, generated deterministically on the central2 TPU | yes; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`, 15625 worlds, 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5/checkpoint.msgpack` | yes; 22656218 bytes, SHA-256 `45a11967333646825f4ac50c47e9e5c9f3b316908aa54af427003c909d65a538`, serialized step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/5sxrd7py` | yes; ID `5sxrd7py`, authenticated state `finished`, last history step 2000 |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5/metrics.jsonl` | yes; 220 records, 292768 bytes, SHA-256 `4fa6f19a0e6b13e8b34906f8a0ce464b9539642f700c2d7b1df73152e3016060` |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5/outputs.jsonl` | yes; 200 diagnostic-cadence records from steps 10 through 2000, 2807042 bytes, SHA-256 `d41dc4c31f7cb876ba4bbc8b437ed883ab3fca9cf2b0d2da3385324978ae360b` |
| Effective configuration | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5/config.yml` | yes; 853 bytes, SHA-256 `5566aadfddb55e1c0dea7d40a7debe03af1f4d9a69f61d6f3341fb25a3e59e01` |
| Canonical log | remote raw log `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5.launcher.log`; deterministic local gzip `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5/20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5.launcher.log.gz` | yes; raw 20201 bytes and SHA-256 `1a1e88caf3c529cce056ae2d0ecd8f3a919bbbccbed60f51fdf8ef1c1e9fcd26`; gzip 4226 bytes and SHA-256 `3bd9836708b76927ba467367ca39bcbbe2a74acb224412f825aeef0863bc6101` |

## Observations

- The application source now falls back to the v4 image's working `gsutil` only when `gcloud storage` is unavailable. Focused regional-data and JAX tests passed before this retry.
- The selected v4-8 runtime, topology, regional dataset, and disk headroom were verified during retry 4 and will be revalidated before launch.
- The fresh mandatory inventory again selected this exact runtime-ready v4-8 from the final `AVAILABLE` table. The remote fixed code returned `BUCKET_LOCATION=US-CENTRAL2`, and the remote source and dataset hashes match their local evidence.
- W&B initialized as run `5sxrd7py`, training reached every expected evaluation boundary, emitted `TRAINING_COMPLETE` at step 2000, and finalized W&B in state `finished` at history step 2000.
- The 199 post-warmup throughput intervals had minimum 4313.790311, median 5822.019561, mean 5806.751187, and maximum 5924.988571 examples/s. The median is 4.306% above the prior v5e-8 baseline median of 5581.683917 examples/s, though this is a hardware-generation comparison rather than a controlled same-device measurement.
- Telemetry contains only compact 32-bin message/receiver activation histograms. No vocabulary-sized token histogram field exists in the 220 metric records, and sampled decoded outputs appear only at the 10-step logging cadence.
- The terminal single-worker audit found the TPU `READY` and `HEALTHY`, the exact trainer absent, all VFIO resources idle, and `/tmp/libtpu_lockfile` absent. The checkpoint payload independently deserialized with `step=2000`.

## Interpretation

- The 2,000-step JAX rerun completed with authenticated W&B tracking and no histogram-size failure.
- Final validation is close to the prior 2,000-step baseline (`0.0390625` exact match versus `0.04296875`; `-4.053215981` mean joint log probability versus `-4.054060936`) and the loss is slightly lower (`3.934626579` versus `3.954409122`).
- The compact histogram design preserves activation-distribution diagnostics without creating the large vocabulary table that broke the earlier W&B attempt.

# Untied width-128 2,000-step run preflight transport attempt 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64-preflight-transport-attempt1` |
| Status | `failed` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261016-conlang-jax-v4-8-n1024-s2000-h128-out256-tied-g64` |
| Git revision | reviewed source and configuration `1f292c854aecfa4b23641736eadcd177c21d7660` |
| Command | `PATH=/snap/google-cloud-cli/current/bin:$PATH bash .agents/skills/launch-elf-training/scripts/preflight_remote_elf.sh he-vision-group kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b us-central2-b /home/kristinelu/elf-data/conlangs-v1 /home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64 1` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host |

## Configuration

Source: `configs/conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64.yaml`

Complete effective configuration: not produced; training did not start. The selected YAML is recorded in the primary run report.

## Changed Hyperparameters

No scientific change from the selected untied arm; this record describes only a read-only preflight transport attempt.

## Results

No training result. Step 1 did not complete and no W&B run or checkpoint was created.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64.yaml` | yes |
| Source snapshot | `/home/kristinelu/elf-runs/source/20261016-conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64` | yes; node-local before preflight |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; node-local and reverified |
| Output | `/home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1024-s2000-h128-out256-untied-g64` | yes; confirmed absent after the failed attempt |
| Checkpoint | none | yes; training never started |
| Tracking run | none | yes; W&B never started |
| Detailed output | not recorded | no; the foreground command returned only connection metadata |

## Failure Details

- Last known step: `0`
- Error summary: the all-worker gcloud SSH preflight returned exit status 1 after connecting, without emitting the remote preflight result or an application error.
- Detailed output: not recorded beyond foreground connection metadata

## Observations

- Direct API and occupancy evidence before the attempt reported `READY`, `HEALTHY`, and idle.
- A direct read-only decomposition after exit found no VFIO holder, no `train.py`, no output directory, no libtpu lockfile, 86 GiB free, and writable `/tmp/tpu_logs`.
- The unchanged idempotent preflight immediately passed on retry and reported `tpu_device=idle`; no repair or workload mutation was performed.

## Interpretation

- The root cause is unconfirmed. Evidence localizes the attempt to transient SSH/preflight transport rather than TPU occupancy, disk, output freshness, runtime, dataset, or application failure.
- Because the attempt was read-only and created no output, W&B identity, trainer, or checkpoint, retrying the same selected run identity did not risk duplicate training.

## Updates

- `2026-10-16T04:38:39Z`: Preserved the zero-step transport attempt, completed the decomposed checks, and confirmed the unchanged preflight passed.

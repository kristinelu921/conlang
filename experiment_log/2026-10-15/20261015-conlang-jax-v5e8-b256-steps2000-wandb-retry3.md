# Conlang JAX v5e-8 batch-256 2,000-step W&B retry 3

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry3` |
| Status | `superseded before staging` |
| Baseline | `20261015-conlang-jax-v5e8-b256-steps2000`; retry of zero-step attempts 1 and 2 |
| Git revision | pending config commit and immutable launch tree |
| Command | pending immutable snapshot path |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, topology `2x4`, expected 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry3.yaml`

The complete configuration is preserved in the committed source file above. It is scientifically identical to retry 2; only the run identity, output path, W&B name, and immutable snapshot identity are fresh.

## Changed Hyperparameters

No scientific or telemetry changes from retry 2. W&B remains enabled. Compact 32-bin activation histograms are logged at `log_every: 10`; vocabulary-sized token histograms and their large table are omitted.

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
| Configuration | `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry3.yaml` | pending commit |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | pending config update |
| Source snapshot | pending | no |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, sourced from `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | previously verified: 7 files, 11643131 bytes, 15625 worlds and 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry3/checkpoint.msgpack` | no |
| Tracking run | pending | no |
| Metrics | pending | no |
| Evaluation outputs | pending | no |
| Canonical log | pending | no |

## Observations

- The required filtered inventory selected the runtime-ready us-west4 Kristine v5e-8 from its final `AVAILABLE` table.
- This fresh run identity preserves both earlier zero-step failures without overwriting them.
- Before staging, the selected v5e-8 was claimed by a healthy current-user LaVin-DiT validation that held all eight VFIO devices for more than 40 minutes. It was identified read-only and left untouched.
- This identity was never staged or launched and created no output directory, launcher log, W&B run, metrics, or checkpoint. The capacity fallback uses the distinct retry-4 identity.

## Terminal Reconciliation

- This planning identity was superseded before staging by retry 4 and is not a
  runnable entry point in the finalized change.
- Its YAML was removed from the finalized configuration closure. The historical
  source remains available in Git commit `d417f52`.
- Every `pending` field above records the preserved pre-launch planning state;
  each is terminally resolved as not applicable because the run never launched.
- Successor evidence is recorded in `20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry4.md`.

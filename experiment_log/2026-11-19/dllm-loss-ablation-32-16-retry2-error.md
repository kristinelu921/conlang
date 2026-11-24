# dLLM loss ablation 32→16 — setup attempt 2

## Failure Record

| Field | Value |
|---|---|
| Timestamp | 2026-11-19 UTC |
| Run ID | `dllm-loss-ablation-32-16-20261119-r2` |
| Retry | 2 |
| TPU | `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, `us-west4-a`, v5litepod-16 topology `4x4`, Spot |
| Stage reached | validation dataset process shutdown after successful artifact write |
| Step 1 completed | no |
| Last metric step | none |
| Verified checkpoint | none; parent localization had not started |
| Artifact mirror | none for the failed attempt |

## Evidence and Recovery

Retry 2 successfully reused the exact pinned environment, enumerated four local TPU devices with process count 1, regenerated all 12,800 FineCode training records in `us-west4`, and reproduced the original training NPZ and manifest SHA-256 values exactly. It then wrote the 64-example validation NPZ and manifest with `training_overlap=0` and data SHA-256 `22fcaefad51954e8bf45152b3ee27d36e1c30652458f0f326a3a945c0d961f2f`.

The validation process remained alive for several minutes after both final files were written. PID `256794`, owned by `kristinelu`, was revalidated as the exact `reproduction.prepare_validation` process. The completed artifacts were checked against their manifest before TERM was sent. The setup command exited `143`; its full log is `/home/kristinelu/elf-runs/staging/dllm-loss-ablation-32-16-20261119-r2/setup.log`.

Retry 3 reuses these verified immutable inputs, localizes and SHA-verifies the parent checkpoint, and launches into the fresh node-local output `output-r3` with the separate regional mirror prefix `retry-r3`.

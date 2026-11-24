# dLLM loss ablation 32→16 — setup attempt 1

## Failure Record

| Field | Value |
|---|---|
| Timestamp | 2026-11-19 UTC |
| Run ID | `dllm-loss-ablation-32-16-20261119-r1` |
| Retry | 1 |
| TPU | `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, `us-west4-a`, v5litepod-16 topology `4x4`, Spot |
| TPU state | `READY`, `HEALTHY`; all four workers were available before setup |
| Stage reached | TPU backend import probe after dependency verification |
| Step 1 completed | no |
| Last metric step | none |
| Verified checkpoint | none; the parent checkpoint was not yet localized |
| Node-local paths | immutable source and exact runtime only; no data, output, or checkpoint |
| Artifact mirror | none for the failed attempt |

## Command and Evidence

The exact retry-1 setup command and full output are preserved at `/home/kristinelu/elf-runs/staging/dllm-loss-ablation-32-16-20261119-r1/setup.log` on worker `t1v-n-652185e1-w-0`.

The managed CPython 3.10.18 environment passed the pinned dependency check and installed W&B 0.22.2. The default JAX TPU probe then warned after 60 seconds that it was waiting for all TPU hosts. The exact stuck process was PID `252850`, owned by `kristinelu`, running only the JAX backend assertion. It completed no setup progress for more than two minutes, held worker 0's four VFIO devices, and was terminated with TERM after its identity was revalidated. The resulting holderless lock was removed through the guarded occupancy helper, and all four workers then reported `AVAILABLE`.

## Diagnosis and Recovery

The selected v5e-16 slice has four hosts, while the reviewed trainer intentionally requires `jax.process_count() == 1`. A bounded compatibility probe with `TPU_PROCESS_BOUNDS=1,1,1`, `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`, and `TPU_VISIBLE_CHIPS=0,1,2,3` succeeded on worker 0 and reported the TPU backend, process count 1, and four local v5e devices.

Retry 2 uses a fresh run/output identity, reuses the exact verified runtime read-only through a node-local symlink, and applies those parameter-free backend bounds during setup and training.

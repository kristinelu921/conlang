# dLLM progressive teacher-weighted 1,600-update retry 0 error

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-progressive-teacher-weighted-1600-20261119` |
| Status | `failed` |
| TPU | `kmh-tpuvm-v5e-16-spot-kristine-19-10-41-03-097732204-0-us-west4-a`, `us-west4-a` |
| Source revision | `35a1613e32afe21316ff2f00cc1a2d1593501de9` |
| Last known step | `0` |
| W&B run | not created |
| Checkpoint | not created |

## Failure

The committed source, region-matched dataset, fixed validation data, and verified 32-step parent checkpoint were localized successfully on worker 0. A bounded four-chip JAX topology probe succeeded, but repeatedly reported that `/tmp/tpu_logs` was not writable. The required launch preflight therefore rejected the candidate before training, output creation, or W&B initialization.

All four workers had a collaborator-owned directory with mode `0775`:

| Worker | Owner | UID | Writable by `kristinelu` |
|---|---|---:|---|
| `t1v-n-c930e01a-w-0` | `sqa` | `2021` | no |
| `t1v-n-c930e01a-w-1` | `sqa` | `2017` | no |
| `t1v-n-c930e01a-w-2` | `sqa` | `2005` | no |
| `t1v-n-c930e01a-w-3` | `sqa` | `2014` | no |

No directory, process, lock, or collaborator-owned state was changed. The localized zero-step attempt remains preserved at `/home/kristinelu/elf-runs/dllm-progressive-teacher-weighted-1600-20261119` on worker 0. Recovery must use a fresh run identity on a worker whose `/tmp/tpu_logs` is absent or writable.

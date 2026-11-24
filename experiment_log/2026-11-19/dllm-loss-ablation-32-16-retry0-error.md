# dLLM loss ablation 32→16 — setup attempt 0

## Failure Record

| Field | Value |
|---|---|
| Timestamp | 2026-11-19 UTC |
| Run ID | `dllm-loss-ablation-32-16-20261119` |
| Retry | 0 |
| TPU | `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, `us-west4-a`, v5litepod-16 topology `4x4`, Spot |
| TPU state | `READY`, `HEALTHY`; all four workers reported `AVAILABLE` before setup |
| Stage reached | dependency setup |
| Step 1 completed | no |
| Last metric step | none |
| Verified checkpoint | none; the parent checkpoint was not yet localized |
| Node-local paths | source and partial environment only; no data, output, or checkpoint |
| Artifact mirror | none for the failed attempt |

## Command and Evidence

Exact command:

```text
bash /home/kristinelu/elf-runs/staging/dllm-loss-ablation-32-16-20261119/setup_remote.sh /home/kristinelu/elf-runs/dllm-loss-ablation-32-16-20261119 /home/kristinelu/elf-runs/staging/dllm-loss-ablation-32-16-20261119 gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119
```

Exit status: `1`.

Detailed log: `/home/kristinelu/elf-runs/staging/dllm-loss-ablation-32-16-20261119/setup.log` on TPU worker `t1v-n-652185e1-w-0`.

Error endpoint:

```text
The virtual environment was not created successfully because ensurepip is not available.
Failing command: /home/kristinelu/elf-runs/dllm-loss-ablation-32-16-20261119/venv/bin/python3.10
```

## Diagnosis and Recovery

The image exposes Python 3.10.12, so the original initializer selected the system-Python branch, but the image lacks the matching `python3.10-venv`/`ensurepip` component. The source archive and dependency lock had already passed SHA-256 verification. No JAX client, W&B run, training process, output, metric, or checkpoint was created.

The failed directory and log are preserved. Retry 1 uses a fresh node-local run identity and the already pinned `uv` 0.8.3 managed-Python 3.10 path, then verifies the exact dependency lock, JAX 0.6.2 TPU backend, and W&B 0.22.2 before data preparation.

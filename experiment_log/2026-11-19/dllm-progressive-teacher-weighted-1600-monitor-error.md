# dLLM 1,600-update monitor infrastructure error

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-progressive-teacher-weighted-1600-20261119`, retry 1 |
| Status | `failed` |
| Failed component | first scheduled local monitor wrapper |
| Training impact | none; trainer remained healthy |

## Failure

At `2026-11-19 16:59:32 UTC`, the scheduled wrapper successfully verified the TPU as `READY`/`HEALTHY`, then exited before its remote inspection because an awk field reference was expanded by the local shell under `set -u` (`$3: unbound variable`). This was a monitor infrastructure error, not a trainer or TPU failure.

The same scheduled check was completed immediately with a corrected read-only command. Worker 0 had the exact expected trainer at update 800, workers 1–3 had no trainer, W&B was `running` at step 799, and the update-700 checkpoint marker was checksum-pinned. No recovery or process action was taken.

Evidence: `experiment_log/2026-11-19/artifacts/dllm-progressive-teacher-weighted-1600/monitor-30m.log`.

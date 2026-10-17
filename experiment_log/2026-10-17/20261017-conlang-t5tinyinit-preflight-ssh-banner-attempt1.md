# T5-tiny warm-start TPU preflight SSH failure

## Run Summary

| Field | Value |
|---|---|
| Attempt | preflight attempt 1; no training retry identity created |
| Intended run | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `failed` before staging |
| Started | `2026-10-17T22:02:00Z` |
| Finished | `2026-10-17T22:03:00Z` |
| TPU | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b` |
| Zone / topology | `us-central2-b` / `v4-8` |
| State / health / scheduling | `READY` / `HEALTHY` / Spot |
| Git revision | `dcc78134e5bb262ddd430dcc72f24c4c002a1d90` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Last metric step | none; Python never started |
| Checkpoint | none |

## Command

The failing read-only command used the standard gcloud all-worker reconnect with
`--internal-ip --ssh-flag=-F --ssh-flag=/dev/null` and a shell body that only
inspected disk, output freshness, runtime imports, the existing node-local dataset,
the pretrained-model cache, `/tmp/tpu_logs`, and matching Python processes:

```text
gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='<read-only preflight body>' --quiet
```

Exit status: `1` after the bounded gcloud retry loop.

## Failure Details

- Stage reached: preflight SSH transport, before source staging, dependency setup,
  data staging, output creation, W&B, JAX, or TPU initialization.
- First error: `kex_exchange_identification: banner line contains invalid characters`;
  `banner exchange: Connection to 10.130.0.93 port 22: invalid format`.
- Detailed log: control-session transcript only; no remote log was created.
- Runtime packages: not inspected because the SSH identification exchange failed.
- Source/data/output/checkpoint locality: not inspected in this attempt; no path was
  created or modified and no checkpoint exists.
- Confirmed scope: failure occurred before authentication while the direct TPU API
  still reported `READY`, `HEALTHY`, `v4-8`, and Spot. The immediately preceding
  canonical occupancy check reported the sole worker `AVAILABLE`.
- Root cause: unconfirmed lower-level port-22 identification-stream failure in the
  gcloud all-worker reconnect path. Authentication, runtime contents, dataset state,
  and model code were not reached and therefore are not causes of this attempt.
- Recovery: use the documented explicit `--worker=0` internal-IP path with bounded
  retry for this single-worker v4-8, then require runtime, W&B authentication,
  dataset, disk, lock, and output-freshness checks before any mutation.
- Verification command: the explicit-worker read-only preflight is pending.

## Artifacts

No source snapshot, remote source, model artifact, output directory, W&B run,
trainer, checkpoint, or launch log was created by this failed read-only attempt.

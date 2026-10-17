# Communication diagnostic artifact-transfer retry

## Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Status | `resolved` |
| Failure scope | post-run artifact transport only; training, W&B finalization, TPU health, and remote artifacts were unaffected |
| TPU | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b` |
| Canonical local archive | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-commdiag/`; checkpoint remains in verified node-local run storage |

## Failure

The first recursive `gcloud compute tpus tpu-vm scp` download intermittently failed with SSH `message authentication code incorrect`, `Bad packet length`, and `Connection corrupted` errors. Its automatic retry returned exit zero but left a complete nested copy plus a separate 425984-byte partial `checkpoint.msgpack`, so that destination was quarantined and never treated as canonical. A subsequent multi-file attempt also returned exit one after transporting some complete files.

## Recovery

- Kept the remote output directory unchanged and rehashed every canonical artifact there.
- Retried files individually over the alternate external-IP SSH route.
- Compared local and remote byte sizes and SHA-256 digests.
- Restored a temporary local checkpoint copy with Flax serialization and verified checkpoint step `2000` plus `state` keys `opt_state`, `params`, and `step`; the validated copy was not retained in Git because run archives intentionally exclude large checkpoints.
- Preserved the canonical launcher log, which contains `TRAINING_COMPLETE` and W&B's successful final sync.

## Verified artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoint.msgpack` | 26418137 | `2596e857c843538e76f18b7325dcecdd281d7a0761818cd5de0372a2e9dbd9c4` |
| `config.yml` | 987 | `4f49fb0bcc8f1c5714e256c9fe4b01a85fc9d11915e94b1af6ffe7407543f2a8` |
| `metrics.jsonl` | 346241 | `e6beab96cb957af0b5b4c436652dd0b8c868b5cd42289d05b2c9f4cd3b8af5c4` |
| `outputs.jsonl` | 2806305 | `997f0d8b6d0bc347edfa1d62ad92f9f0ecae2faca53ee1a18d1f074fdc4143ee` |
| remote `launcher.log` | 18656 | `8495eaf04d42be679f4fb15b6b89dff17af735818a63925ab6094e8603d8e30a` |

## Impact

None on experiment results. The transport error occurred after training and final evaluation. The canonical archive is complete and hash-identical to the TPU copies.

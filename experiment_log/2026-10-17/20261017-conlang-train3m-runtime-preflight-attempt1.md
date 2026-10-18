# Train-3M runtime preflight attempt 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-train3m-runtime-preflight-attempt1` |
| Status | `failed` |
| Started | `2026-10-17T22:57Z` |
| Finished | `2026-10-17T22:57Z` |
| Baseline | `not specified` |
| Git revision | `1f69e2243f4508bb14fdef56ff2a47d525d38a2f` |
| Command | exact command below |
| Seed | `not applicable` |
| Hardware | selected `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; no cloud call was reached by this attempt |

## Command

```bash
PATH=/snap/google-cloud-cli/current/bin:/usr/local/bin:/usr/bin:/bin ELF_NETRC_PATH=/kmh-nfs-ssd-us-mount/code/kristine/conlang/.netrc ELF_TPU_INITIALIZE_MISSING=1 bash tpu_scripts/initialize_elf_runtime.sh he-vision-group kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b us-central2-b
```

## Configuration

Source: `not applicable; control-host runtime-readiness preflight`

The intended training configuration was
`configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml`,
but the initializer stopped before its first cloud call and never staged or
launched that run.

## Changed Hyperparameters

No confirmed changes; no training process was created.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| not applicable | initializer exit status | `1` | `0` | not applicable | control-host stderr |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Failed credential path | `/kmh-nfs-ssd-us-mount/code/kristine/conlang/.netrc` | verified absent |
| Correct credential path | `/home/kristinelu/.netrc` | verified owner `kristinelu`, mode `0600`; contents not printed |
| Run output | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` | not created or contacted by this local-only attempt |
| Checkpoint | not applicable | no |
| Tracking run | not applicable | no |

## Failure Details

- Last known step: 0; training was not contacted or launched.
- Stage reached: local credential-path preflight.
- Error summary: `ELF W&B credential file is missing: /kmh-nfs-ssd-us-mount/code/kristine/conlang/.netrc`.
- Exit status: `1`.
- Source/data/output/checkpoint locality: not reached.
- Detailed output: control-host command output; no separate log was created.

## Observations

- The isolated launch worktree intentionally excludes credentials, and the
  assumed main-repository credential path was also absent.
- Because the initializer validates its credential path before any `gcloud`
  command, this attempt made no cloud call, remote write, W&B run, output,
  checkpoint, or TPU-runtime change.
- The verified repair used `/home/kristinelu/.netrc`. The otherwise identical
  command then reported `ELF_RUNTIME_READY` and
  `RUNTIME_INITIALIZATION=REUSED` for the selected worker.

## Interpretation

- Root cause: the caller supplied a stale repository-relative credential path
  instead of the established owner-only home-directory credential.
- Rejected alternatives: TPU health, occupancy, runtime packages, and W&B
  authentication were not implicated; the repaired readiness command passed.
- Recovery: retain the original scientific run identity because this attempt
  stopped before cloud access and could not create any run-scoped state.

## Updates

- `2026-10-17T22:58:20Z`: Recorded the zero-step local preflight failure and
  the successful credential-path correction.

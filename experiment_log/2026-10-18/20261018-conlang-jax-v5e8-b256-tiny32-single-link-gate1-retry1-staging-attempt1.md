# Tiny single-link gate retry 1 source-staging attempt

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry1` |
| Attempt | source-staging attempt 1 |
| Status | `failed` before launch |
| Started | `2026-10-19T02:44:05Z` |
| Finished | `2026-10-19T02:45:30Z` |
| Git revision | `887ac10` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; READY/HEALTHY and VFIO idle |

## Stage and Command

Stage reached: source transfer. The corrected local snapshot excluded `.git`
and verified as 72 files / 528,415 bytes with relative-manifest SHA-256
`060ccaf498e96fc3689ba88ea5c717f2c9ca01988312712945d52d45061272fb`.

The remote timestamped target was created fresh and remained empty. The copy
command was:

```bash
gcloud compute tpus tpu-vm scp --recurse \
  /kmh-nfs-ssd-us-mount/staging/kristine/elf/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry1-20261019T024405.095745038Z/. \
  kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a:/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-retry1-20261019T024405.095745038Z/ \
  --project=he-vision-group --zone=us-west4-a --worker=all
```

Every bounded retry returned:

```text
scp: error: unexpected filename: .
```

The local retry loop was interrupted after the repeated deterministic error;
the foreground command exit status was `130`.

## Results

| Check | Result |
|---|---|
| Local snapshot | corrected and fully verified |
| Remote source target | exists but contains zero entries |
| Regional dataset | seven verified us-west4 objects; node-local copy not started |
| Run output / launcher log | absent |
| Trainer / W&B / checkpoint | none |
| TPU holder | none |

## Failure Details

- Last known step: none; Python training never started.
- Error summary: OpenSSH SCP rejected the literal recursive source basename
  `.` before transferring a file.
- Detailed output: concise repeated stderr is preserved above; no remote
  launcher log exists.
- Packages before / after: unchanged runtime JAX `0.6.2`, Flax `0.10.5`, Optax
  `0.2.8`, and W&B `0.28.1`.
- Checkpoint: none.

## Root Cause and Recovery

### Recursive SCP rejects a dot source basename

- Attempt / retry: gate retry 1, source-staging attempt 1.
- TPU / zone / topology: selected us-west4-a Spot v5e-8 above.
- Stage reached and last step: source transfer; no metric step.
- First error and exit status: `scp: error: unexpected filename: .`; local
  retry loop stopped with `130`.
- Confirmed root cause and evidence: passing the immutable directory as
  `<snapshot>/.` makes current SCP treat `.` as an unsafe unexpected filename;
  the same error repeated before any remote entry appeared.
- Alternative causes rejected: exact SSH access and directory creation worked;
  free disk was 4 GiB; TPU remained idle; the fresh target remained empty.
- Repair command: build a task-local gzip tar archive from the immutable
  snapshot, record its SHA-256, transfer that regular file, verify the same hash
  remotely, and extract it into a new fresh timestamped directory.
- Verification command: require the extracted relative-manifest SHA-256 and
  selected-config SHA-256 to match the trusted local values before launch.
- Checkpoint / artifact state: no checkpoint; empty remote target retained as
  evidence and never reused.
- Recovery action: stop the local SCP retry loop and select a fresh run/output/
  W&B identity before archive-based staging.
- Prevention gate added: the error catalog now rejects `<snapshot>/.` as a
  gcloud SCP source and requires a hashed regular-file archive or another
  transfer whose exact remote directory semantics are verified.

## Updates

- `2026-10-19T02:45:30Z`: Preserved the deterministic zero-step transfer
  failure and verified that no scientific or accelerator state was created.

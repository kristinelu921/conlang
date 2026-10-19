# Tiny single-link gate source-staging attempt 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1` |
| Attempt | source-staging attempt 1 |
| Status | `failed` |
| Started | `2026-10-19T02:37:46Z` |
| Finished | `2026-10-19T02:42:11Z` |
| Git revision | `2eb576e5f4040d98e0546605655729002251c6f6` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; READY/HEALTHY and VFIO idle |

## Stage and Commands

Stage reached: source staging. No training process, W&B run, output directory,
metric step, or checkpoint was created.

The local snapshot command succeeded and returned the path below:

```bash
ELF_STAGING_SOURCE=/tmp/conlang-tiny32-single-link-launch \
ELF_STAGING_CONFIG=configs/conlang-jax-v5e8-b256-tiny32-single-link-gate1.yaml \
bash tpu_scripts/run_staging.sh \
  20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1
```

The transfer command exited `0` but copied snapshot contents directly into the
already existing remote `sources/` directory rather than creating the expected
timestamped child:

```bash
gcloud compute tpus tpu-vm scp --recurse \
  /kmh-nfs-ssd-us-mount/staging/kristine/elf/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-20261019T023746.146739237Z \
  kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a:/home/kristinelu/elf-runs/sources/ \
  --project=he-vision-group --zone=us-west4-a --worker=all
```

The first exact-path verification failed its opening `test -d` because this
directory was absent:

```text
/home/kristinelu/elf-runs/sources/20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-20261019T023746.146739237Z
```

The task-created misplaced tree was moved intact, not deleted, to:

```text
/home/kristinelu/elf-runs/sources-misplaced-20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1-attempt1
```

## Configuration

The planned attempt used the immutable configuration recorded in
`20261018-conlang-jax-v5e8-b256-tiny32-single-link-gate1.md`. No configuration
field reached model construction.

## Results

| Check | Result | Evidence |
|---|---|---|
| TPU API | READY, HEALTHY, Spot | exact-resource describe |
| Occupancy | `AVAILABLE`; no VFIO holder | guarded occupancy helper and remote preflight |
| Local snapshot | 72 files, 526,870 bytes | local snapshot audit |
| Expected remote child | absent | fresh SSH exact-path check |
| Remote `sources/` root | contained the snapshot top-level entries directly | depth-one remote inventory |
| Training step 1 | not reached | no trainer, log, output, W&B run, or checkpoint |

## Failure Details

- Last known step: none; Python training never started.
- First error: required remote timestamped source directory absent after an
  apparently successful SCP command.
- Transfer exit status: `0`; exact-path verification failed logically at its
  first `test -d`.
- Detailed log path: command output preserved in this report; no launcher log
  exists because launch did not begin.
- Checkpoint: none.
- Data: seven objects totaling 11,992,785 bytes were independently verified in
  `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-v1`; node-local
  localization had not begun.
- Output: exact run output and launcher-log paths remained absent.

## Root Cause and Recovery

### Worktree `.git` file and SCP destination semantics

- Attempt / retry: source-staging attempt 1.
- TPU / zone / topology: selected us-west4-a Spot v5e-8 above.
- Stage reached and last step: source staging; no metric step.
- Confirmed root cause and evidence: `run_staging.sh` excluded `.git/`, which
  does not exclude the `.git` file used by a linked Git worktree. The immutable
  local snapshot therefore contained that pointer file. Independently, recursive
  SCP to an existing `sources/` directory placed the snapshot contents directly
  in that directory; the required timestamped child was absent while the exact
  snapshot top-level inventory appeared at the parent.
- Alternative causes rejected: the TPU remained READY/HEALTHY and idle; SSH
  and SCP completed; remote disk reported 4 GiB free; runtime imports passed;
  no dataset, output, trainer, W&B run, or checkpoint existed.
- Packages before / after: unchanged runtime JAX `0.6.2`, Flax `0.10.5`, Optax
  `0.2.8`, and W&B `0.28.1`.
- Repair command: change the rsync exclusion to `--exclude='.git'`; copy the
  next immutable snapshot into a pre-created fresh timestamped directory using
  `snapshot/.` as the source.
- Verification command and result: `python -m pytest -q tests/test_staging.py`
  passed `1` test; the next snapshot must contain no `.git` entry and its
  relative file-manifest SHA-256 must match after node-local transfer.
- Checkpoint / artifact state: no checkpoint; failed local snapshot and remote
  quarantine retained as immutable evidence.
- Recovery action: moved the misplaced remote tree intact to the run-scoped
  quarantine path and recreated an empty `sources/` parent.
- Prevention gate added: a focused staging fixture now models a worktree-style
  `.git` file and requires it to be absent from the snapshot.

## Updates

- `2026-10-19T02:42:11Z`: Recorded the zero-step staging failure, preserved the
  local snapshot and remote tree, and verified the focused exclusion repair.

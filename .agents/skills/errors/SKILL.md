---
name: errors
description: Diagnose, prevent, and document ELF TPU staging, dependency, distributed-launch, training-startup, throughput-regression, checkpoint, and monitoring failures. Use whenever an ELF run fails, stalls, or materially regresses before or after launch, a TPU candidate fails preflight, a runtime package is missing or incompatible, staging/data verification fails, a Spot TPU is preempted, or an experiment report needs an immutable error postmortem.
---

# ELF TPU Launch Errors

Treat this file as the singular living catalog for ELF staging and launch failures. Preserve observed evidence, distinguish confirmed causes from hypotheses, and append any new failure class here after recording the concrete attempt under `experiment_log/`.

## Mandatory error record

For every failed staging or launch attempt, immediately create or update a distinct immutable experiment report and record:

- UTC timestamp, run ID, retry number, TPU name, zone, topology, state, health, and Spot flag.
- Stage reached: discovery, preflight, source staging, dependency setup, data staging, import, distributed initialization, model construction, optimizer construction, compilation, first step, checkpoint, or monitoring.
- Exact command and exit status.
- Concise error and detailed log path; retain only a short traceback endpoint in the report.
- Installed and required package versions for every dependency failure.
- Last confirmed metric step, whether step 1 completed, and newest verified checkpoint or `none`.
- For throughput failures, the global microbatch, training-loop step count, interval wall time, accumulation count, examples/second, excluded non-steady-state intervals, and accepted benchmark/tolerance.
- Whether source/data/output/checkpoints were node-local and whether any artifact mirror was verified.
- Root cause with evidence, rejected alternative causes, repair command, verification command/result, and recovery action.

Never collapse a failed zero-step attempt into a later successful retry. Use a new output path, mirror prefix, and report for every retry. Never overwrite a prior record.

## Triage order

1. Preserve the foreground SSH exit status and physical worker-0 log tail as
   initial evidence, not as the terminal verdict.
2. Query the TPU with the server-side `kristine` name filter; record `READY`, health, Spot, and current existence.
3. Check every worker for the training PID, numeric UID, `/dev/vfio/0` holder,
   newest log timestamp, completion/failure sentinel, last metric step, disk,
   and checkpoints; identify the physical process-zero log worker from evidence.
4. Classify the failure by the last completed stage below.
5. Do not blindly retry a deterministic failure. Repair and verify it without opening the TPU device when possible.
6. Resume only from a checkpoint whose existence, size, and integrity were verified. Otherwise record `no valid checkpoint` and restart the exact configuration.
7. Never terminate a holder except through the guarded occupancy-helper reclaim allowed on an exact authorized `kristine` TPU for a positively identified foreign, non-`kristine` VFIO holder, the verified stuck current-user ELF case on that same TPU, or an explicit user authorization to terminate a process owned by kristine. A process owned by kristine may be terminated only when the user explicitly authorizes that termination, and never as part of a sweep. On a non-`kristine` TPU, do not edit, stop, or touch an actively held run owned by somebody else at any point, even when the user requests it. Never interrupt, affect, or make yourself known to a run on a non-`kristine` TPU. A later full sweep may read-only re-query a TPU previously found actively held, because its job may have finished and the TPU may be inactive again. Every sweep stays read-only and must never touch an existing active job. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it. On a `kristine` TPU, the documented recovery rules apply. Never reboot a TPU or delete a resource as recovery. A non-`kristine` TPU may be used only after `AUTHORIZE WIDE READ-ONLY TPU SWEEP` and, for that exact idle resource, `AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION`.

## Discovery and capacity failures

### No authorized TPU found

- Symptom: filtered inventory returns no `kristine` resource.
- Cause: no authorized capacity exists or Cloud Asset Inventory lags.
- Verify: direct TPU API list with server-side `name:kristine` in known zones.
- Recovery: keep the filtered `kristine` audit on the monitoring cadence. A wide read-only sweep of other TPUs requires `AUTHORIZE WIDE READ-ONLY TPU SWEEP` and must not touch an active run. Run continuity may also use an equivalent `kristine` Spot replacement.

### `CREATING` never becomes usable

- Symptom: TPU remains `CREATING`, disappears, or reports an internal create error.
- Causes: transient Spot capacity, provisioning failure, quota/rate limits, invalid service account, or nonviable zone.
- Verify: original create operation plus direct filtered inventory; do not count Cloud Asset alone.
- Recovery: keep one orchestrator, account for unresolved creates, and avoid duplicate campaigns.

### `no more capacity`, quota, or reservation error

- `no more capacity`: zone lacks current Spot capacity; quota may still exist.
- preemptible quota `0`: retries cannot succeed until quota changes.
- `Reservation not found` on Spot: zone/configuration is not viable.
- create-request rate limit: the orchestrator is retrying too aggressively.
- Recovery: preserve the original campaign, back off, and do not increase topology/count without authorization.

### Cloud Asset `NOT_FOUND` race

- Symptom: asset search returns a TPU that direct describe reports as missing.
- Cause: lagging Cloud Asset data after preemption/deletion.
- Recovery: treat the direct TPU API as authoritative and record the disappearance.

## Ownership and preflight failures

### Live v4 libtpu holder is absent from VFIO inspection

- Symptom: `/dev/vfio/*` has no `fuser` holder, but JAX reports the TPU is in use by a live PID and `fuser /tmp/libtpu_lockfile` identifies that process.
- Cause: a v4 libtpu workload can hold `/tmp/libtpu_lockfile` without appearing as a VFIO holder.
- Recovery: inspect both VFIO resources and `/tmp/libtpu_lockfile`; treat any live lockfile holder exactly like an occupied device and never initialize, claim, or launch on it.

### Candidate is busy

- Symptom: `sudo lsof -w /dev/vfio/0` returns any PID, with or without `train.py`.
- Cause: an accelerator workload already owns the device.
- Recovery: reject the TPU when the holder belongs to the current remote user, is a `kristine` workload, or cannot be positively classified. Only on an exact authorized `kristine` TPU, the occupancy helper may reclaim a positively identified foreign, non-`kristine` VFIO holder during candidate claiming. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it.

### VFIO occupancy changes after preflight

- Symptom: initial preflight reports the TPU idle, but the final all-worker check immediately before the foreground launch finds a `/dev/vfio/*` or live `/tmp/libtpu_lockfile` holder.
- Cause: another workload claimed the TPU during the staging-to-launch time-of-check/time-of-use window.
- Recovery: fail closed before starting any worker, preserve the attempt as a zero-step failure, and record the new holder's worker, PID, numeric UID, and command without terminating it in this final gate. Any permitted foreign-holder reclaim must occur separately through the guarded candidate-claim workflow on an exact authorized `kristine` TPU. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it.
- Prevention: after source, runtime, and data setup, rerun the occupancy helper on every worker and inspect both VFIO and `/tmp/libtpu_lockfile` immediately before opening the foreground fan-out. Require every worker to remain idle; never reuse the earlier preflight result. Implement the final gate as a dedicated read-only occupancy check; never reuse a setup preflight that can initialize, delete, move, or otherwise repair state.

### Exact-target claim guard exits on a terminated zombie

- Symptom: the initial and final all-worker occupancy checks are clean, but JAX then reports `TPU is already in use`; the separately authorized exact-target guard logged `TERM` followed by `REFUSED_RACE`, and revalidation showed the signaled process as `<defunct>`.
- Cause: `kill -0` remains successful for a zombie until its parent reaps it, while `ps` changes the command name to `<defunct>`. Treating that normal terminal state as PID reuse makes the guard exit, allowing an external controller to relaunch the exact foreign job during the final preflight-to-backend-open interval.
- Recovery: preserve the attempt as a zero-step failure and use a fresh retry identity. On an exact authorized `kristine` TPU, the guarded candidate-claim workflow may continue. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it.
- Scope: foreign-holder reclaim applies only to an exact authorized `kristine` TPU. On a `kristine` TPU, the documented recovery rules apply.
- Observed: the first v6e-32 fused linear-mask gate completed immutable staging and two clean occupancy checks, but its guard exited when Zak's signaled `main.py` became a zombie. Zak's external controller reacquired all workers before JAX initialization, so the attempt exited on every worker before step 1 without producing a model, runtime, or throughput verdict.

### Exact-target claim guard skips whitespace-separated `fuser` PIDs

- Symptom: the bounded guard remains live with an empty log while the occupancy
  helper repeatedly reports the same foreign PID on every VFIO resource.
- Cause: `fuser` can emit several space-separated PID entries on one line and
  may omit the final newline. A line-oriented reader either rejects the
  combined value as nonnumeric or skips the unterminated final token, silently
  recording no holder.
- Recovery: keep the launch blocked, preserve the guard PID/log and occupancy
  evidence, split the captured `fuser` value with shell whitespace tokenization,
  and rerun a focused source test before starting a fresh bounded guard.
  Revalidate the exact owner and command fragment before any signal.
- Prevention: require the guard to tokenize, numerically validate, and dedupe
  PID output before owner/start-time/command checks; after startup, prove the
  guard is live and that a known reacquired holder produces a guard action or a
  fail-closed refusal.
- Observed: the FineWeb three-epoch continuation handoff saw PID `654006`
  repeated once per v6e VFIO resource. The original guard treated the entire
  whitespace-separated line as one invalid PID and left the foreign MiniARC
  runner untouched while its own process stayed live.

### Setup activity or another user's process

- Symptom: Python/JAX, setup scripts, tmux/screen, or repository work appears under another numeric UID.
- Cause: concurrent initialization or workload claim.
- Recovery: reject the candidate; a matching TPU name is not proof of idleness.

### TPU not `READY`/`HEALTHY`

- Symptoms: `TIMEOUT`, unhealthy, preempted, deleting, or blank health that persists.
- Causes: provisioning, maintenance, runtime health degradation, or Spot reclaim.
- Recovery: do not reboot. Recheck briefly for initially blank health; otherwise use equivalent authorized replacement recovery.

### Insufficient disk

- Symptom: preflight free-space comparison fails, extraction reaches `ENOSPC`, or checkpoint writes fail.
- Causes: dataset/archive coexistence, model cache, pip cache, crash dumps, stale outputs, or underestimated checkpoint retention.
- Prevent: budget full extracted data, transfer archive/cache, source, logs, compiler artifacts, and at least two checkpoints on every worker.
- Recovery: remove only known recoverable staging artifacts created by the current attempt; never delete unrelated outputs without authorization.

### TPU VM `/dev/shm` is session-scoped

- Symptom: a background dataset builder completes and logs a valid manifest, but its completed output beneath `/dev/shm` is absent on the next SSH connection while durable `/home` logs remain.
- Cause: this TPU VM image may give the SSH workload a private `/dev/shm` mount namespace whose contents disappear when that session's process tree exits.
- Recovery: treat the completed `/dev/shm` output as unavailable. Rebuild into a durable node-local `/home/$USER/elf-data/...` path (after a free-space gate), keeping only regenerable source downloads and caches in `/dev/shm`.
- Prevention: never place a dataset, checkpoint, source snapshot, or other artifact needed by a later SSH session under `/dev/shm`; verify its path remains present from a fresh SSH session before counting it as staged.
- Observed: the official ELF-B/M/L distribution study initially verified all
  75 OWT Arrow shards and `37,384,191,384` canonical bytes under `/dev/shm` on
  eight v4-64 workers. A later fresh all-worker preflight found the path wholly
  absent everywhere before launch, so the retry stopped at zero sequences and
  restaged from the region-matched Central2 bucket to durable node-local
  `/home/kristinelu/elf-data`.

### Complete node-local dataset counted twice

- Symptom: data preflight demands space for the entire remote dataset even though an equal-size node-local dataset with both metadata files already exists.
- Cause: the launcher budgets the full transfer without accounting for reusable local files.
- Recovery: verify the local aggregate file size and required metadata against the remote aggregate size, then require only the post-staging reserve. For an incomplete copy, budget the missing bytes plus the reserve before resuming the sync.

### Output path already exists

- Symptom: launcher refuses to overwrite the exact configured run output.
- Cause: duplicate launch, stale retry, or reused run ID; an existing containing parent alone is not a collision.
- Recovery: inspect and record the existing attempt; preserve it, then select the first output identity absent from every worker (`_v2`, `_v3`, and so on) instead of deleting or reusing the run output.
- Prevention: resolve the effective identity once before source staging. Use that same identity for local and remote source snapshots, the node-local output, the effective training config, W&B, logs, PID files, and the checkpoint mirror prefix; retain a final exact-path freshness gate for concurrent-launch races. Do not create the run output or start its checkpoint mirror until every preflight succeeds.
- Cleanup boundary: tag current-invocation source and empty-output claims with an unguessable ownership value. Before training fan-out, an exit trap may remove only paths whose marker still matches and a local snapshot created by that same invocation. Never roll back shared runtime/data, pre-existing paths, or any artifact after a worker could have started training or produced a checkpoint.

### Fresh VM lacks the output parent

- Symptom: preflight reports the configured node-local output parent missing or unwritable even though the exact run output does not exist.
- Cause: writability was tested before the launcher idempotently created the exact configured parent on a fresh VM.
- Recovery: preserve the attempt as a zero-step failure and use a fresh retry identity. Verify every worker has no trainer, exact run output, or checkpoint; after validating locality, let the launcher create only the parent, never the run output itself, then verify ownership and writability before repeating the output-freshness gate.
- Prevention: treat parent readiness and run-output freshness as separate checks: create the exact node-local parent before testing writability, while requiring the configured run output to remain absent.

## Source and data staging failures

### Timestamped local snapshot collides with a stable remote run path

- Symptom: all workers connect successfully but report `Command execution on
  worker ... failed with exit status 1` while staging source. The exact remote
  command ends in `test ! -e <remote-source-directory>`.
- Cause: the local immutable snapshot basename includes a timestamp, while the
  remote source directory is derived from only the stable config `run_id`.
  Repeating staging therefore maps a new local snapshot onto an existing remote
  directory, and the intentional non-overwrite test returns 1 everywhere.
- Recovery: verify the existing remote snapshot before deciding whether to
  reuse it. For a new copy, preserve the timestamped snapshot basename in the
  remote path or use a distinct run identity. Never delete or overwrite an
  existing source until every active run has been shown independent of it.
- Prevention: log the complete failing remote command, distinguish remote
  status 1 from SSH status 255, and require local and remote immutable snapshot
  identities to have the same uniqueness semantics.
- Observed: the FineWeb v5e-32 repeat at `2026-08-12T02:27Z` found the same
  run-ID-only source directory on all eight workers; it had been populated at
  approximately `01:37:53Z` and contained `train.py` everywhere.

### Legacy launcher rejects a valid timestamped existing snapshot

- Symptom: preflight and dataset reuse succeed, then the launcher exits `2`
  with `existing staging directory must end with run_id` before copying source.
- Cause: an older launcher requires the existing snapshot basename to equal the
  stable run ID exactly, while the current staging helper appends a timestamp
  to preserve immutable uniqueness.
- Recovery: preserve the timestamped snapshot and the zero-step attempt. Use
  the immutable source's matching legacy staging helper to create a second
  snapshot whose basename is exactly the run ID, then require the entrypoint
  and selected-config hashes to match before a fresh full preflight. Never
  rename, overwrite, or delete the timestamped evidence.
- Prevention: resolve the existing-snapshot basename contract from the exact
  staged launcher before creating the launch snapshot; record both snapshot
  paths when compatibility requires two byte-identical copies.
- Observed: PG19 retry24 passed every v4-8 preflight and reused the exact
  37,799,932,705-byte dataset, then its retry22-derived launcher rejected the
  timestamp-suffixed snapshot. An exact-run-ID snapshot with matching
  `train.py` and config SHA-256 values passed the next launch.

### New launcher requires an optional module absent from a legacy snapshot

- Symptom: immutable source and data staging complete, but the VM-local CPU import probe exits with `ModuleNotFoundError` for a module that the staged training entrypoint does not import.
- Cause: the mutable outer launcher applies dependency assumptions from the current repository to an older pinned source snapshot, so the staging contract is stricter than the actual staged program.
- Recovery: preserve the attempt as a zero-step failure and use a distinct retry identity. Derive optional dependency probes from the immutable staged entrypoint; keep the import mandatory when that entrypoint imports it, and do not inject a current module whose transitive API may be incompatible with the legacy source.
- Prevention: test both a legacy minimal entrypoint without the optional module and a current entrypoint that imports it. Snapshot validation must follow the staged program, not the launcher's repository revision.
- Observed: trial 21's trial-19-derived source copied successfully to a v6e-8 but lacked `evaluation.py`. Its legacy `train.py` did not use evaluation; the current launcher's unconditional evaluation import failed before W&B, TPU initialization, or step 1. Copying the current evaluator was rejected because it required a newer encoder symbol absent from the snapshot.

### Module launch cannot resolve a repository-relative asset

- Symptom: `python -m <package>` imports from the immutable `PYTHONPATH` and
  initializes distributed JAX, then every worker fails to open a relative
  configuration or prompt path even though that file exists in the snapshot.
- Cause: `PYTHONPATH` controls module imports but does not change the caller's
  working directory. Passing a repository-relative path directly to `open()`
  therefore resolves it from the SSH default directory.
- Recovery: preserve the zero-step attempt and use a fresh retry identity.
  Resolve packaged repository assets from the entrypoint's `__file__` root and
  test the module from an unrelated working directory; do not rely solely on a
  launcher-side `cd` to conceal the API bug.
- Observed: all eight retry-3 B distribution processes initialized JAX and
  reported distinct process indices before failing on
  `configs/training_configs/train_owt_ELF-B.yml`. The output path remained
  absent, and the corrected path-resolution test passed from a temporary
  directory before retry.
- Observed again: retry 4 resolved that top-level YAML but failed on its nested
  repository-relative `configs/sampling_configs/uncond_sampling_configs.yml`.
  The prevention test now loads the real top-level and nested configuration
  from an unrelated working directory and verifies that the caller directory is
  restored.

### Source archive cannot be built

- Symptoms: permission denied, unreadable file, archive exits nonzero.
- Causes: credential files, unrelated monitor files, special files, or ownership mismatch in a dirty worktree.
- Recovery: reject the partial archive. Exclude only credentials, caches, experiment logs, environments, outputs, and unrelated unreadable artifacts; record every exclusion and rebuild/hash.

### SCP/SSH interruption

- Symptoms: connection reset, partial file, nonzero fan-out status, or missing worker copy.
- Causes: Spot reclaim, SSH timeout, network interruption, worker unavailability, or insufficient disk.
- Recovery: never assume success from quiet output. Verify file size/hash on every worker; use a replacement if the node disappeared.

### `gcloud` SCP bundles the SSH config flag incorrectly

- Symptom: TPU VM SCP rejects an SSH option or config path, interprets `-F /dev/null` as one malformed argument, or fails because the image's SCP endpoint requires the legacy protocol.
- Cause: each `--scp-flag` forwards exactly one argument; bundling `-F` and `/dev/null` into one value does not preserve their two-argument OpenSSH form, and modern SCP defaults to SFTP unless `-O` is supplied.
- Recovery: pass the options exactly as `--scp-flag=-F --scp-flag=/dev/null --scp-flag=-O`. Preserve a nonzero staging attempt, then verify exit status plus the expected file size or hash on every worker before continuing.
- Prevention: keep all three values as separate repeated `--scp-flag` arguments in reusable staging commands and tests; never use `--scp-flag="-F /dev/null"`.

### Inline remote awk field expands under `set -u`

- Symptom: every worker exits before staging with an error such as `bash: $1: unbound variable`, even though `$1` was intended to be an awk field reference.
- Cause: nested local/remote quoting leaves an awk program in remote double quotes, so the remote shell expands `$1` as a positional parameter before awk runs.
- Recovery: preserve the zero-step attempt, use a distinct retry identity, and run a checked remote script body whose awk program is single-quoted or whose dollar signs are proven literal. Verify that no partial data, output, or checkpoint path was created before retrying.
- Prevention: syntax-check reusable scripts locally and avoid embedding awk programs inside multiply nested inline `--command` strings.

### Multiline remote command is flattened into escaped text

- Symptom: TPU SSH reaches the worker, then Bash reports fused tokens such as
  `/dev/nullntest`, `thenn`, or another word containing the preceding line's
  final character plus the next line's text; no intended staging path exists.
- Cause: a multiline remote body was serialized as an escaped string inside a
  locally assembled `gcloud ... --command` argument. The transport preserved
  escape characters instead of passing the body as one parsed shell program.
- Recovery: preserve the zero-step attempt, verify every exact target and
  temporary path is absent, and verify no trainer or VFIO holder exists. Retry
  only with a shell-quoted single-line command or a staged, syntax-checked
  script whose hash is recorded.
- Prevention: test the exact rendered `--command` argv before any mutation;
  never assume JSON or language-string newline escaping is equivalent to shell
  quoting across the local shell, gcloud, SSH, and remote shell boundaries.
- Observed: checkpoint-2800000 localization attempt 1 exited `127` with
  `/dev/nullntest` and `thenn`; a direct read-only audit confirmed the target,
  temporary download, marker probe, trainer, and W&B run were all absent.

### Deferred checkpoint-localization variable expands in the remote script writer

- Symptom: launch passes source, dataset, topology, runtime-signature, and final
  VFIO gates, then every remote launch shell exits under `set -u` with an error
  such as `resume_temporary_path: unbound variable` before downloading the
  checkpoint or starting the trainer.
- Cause: the localization fragment crosses two shell-rendering boundaries: the
  local launcher builds a remote `printf` command, and that command writes the
  worker-local training wrapper. Escaping a deferred variable for only the first
  construction allows the outer remote shell to expand it before the wrapper's
  assignment runs.
- Recovery: preserve the zero-step output and immutable source, stop only an
  exact run-specific mirror helper after owner/PID/full-command verification,
  and retry under a fresh identity after repairing every deferred parameter,
  command substitution, PID expansion, and awk field reference. Do not weaken
  `set -u` or skip checkpoint size/SHA verification.
- Prevention: extract the production localization fragment, render it through
  both shell layers under `set -u`, syntax-check the resulting wrapper, execute
  it against a fake copy client, and require exact bytes, SHA-256, atomic final
  path publication, and no partial file.
- Observed: the v5e-16 self-prefix retry1 passed all final prelaunch gates and
  created only launcher bookkeeping plus one mirror helper before all four
  workers failed on `resume_temporary_path`. No trainer, W&B run, RAM checkpoint,
  or model checkpoint existed; the exact mirror helper was terminated with TERM
  and no KILL.

### Archive hash mismatch

- Symptom: remote SHA-256 differs from trusted source.
- Causes: partial transfer, wrong artifact, corruption, or source changed during packaging.
- Recovery: do not extract. Record both hashes, remove only the bad staging copy, rebuild or recopy once.

### Dataset metadata or shards missing

- Symptoms: absent `state.json`, `dataset_info.json`, or expected Arrow shard count; `load_from_disk` fails.
- Causes: incomplete extraction/download, wrong directory nesting, wrong dataset format, or partial worker staging.
- Recovery: reject launch; verify exact dataset root, metadata, shard count, size, and free disk on every worker.

### Conditioned object-admin binding blocks a regional dataset seed

- Symptom: both `gcloud storage rsync` and explicit `gcloud storage cp` fail before transferring any object with `403 storage.objects.get` on the destination prefix, even though the TPU service account has a conditioned `roles/storage.objectAdmin` binding.
- Cause: the binding condition does not authorize the destination object reads that the transfer client performs before upload; the role name alone is not proof that its condition matches those requests.
- Recovery: verify that no partial objects were written. If the complete canonical dataset is independently verified on node-local storage, set `dataset_gcs_path: null` for that run so launch does not enter the unusable regional sync path; otherwise reject launch until a readable region-matched source exists.
- Prevention: test destination list/GET and a harmless seed object under the exact service-account condition before depending on a new regional prefix. Require the launcher field to reflect the verified transport actually used.
- Observed: a us-east1 OpenWebText seed failed identically through recursive rsync and an explicit metadata-plus-Arrow copy, leaving the prefix empty while the node-local snapshot retained 78 canonical files totaling `37,384,191,384` bytes.

### `gsutil` fails while importing `OpenSSL.crypto`

- Symptom: `gsutil` exits before a GCS operation with an `AttributeError` such as missing `GEN_EMAIL` while importing `OpenSSL.crypto`.
- Cause: the TPU image's system `pyOpenSSL` package is incompatible with its OpenSSL library, even though the Google Cloud CLI's isolated runtime remains usable.
- Recovery: use a working `gcloud storage ls` to establish exact remote bytes. Prefer `gcloud storage rsync`; when a legacy SDK exposes `gcloud storage cp` but not `rsync`, copy with a recursive current-level glob so nested dataset directories remain intact. Verify metadata, exact bytes, and absence of temporary files on every worker afterward. The launcher must not require a successful `gsutil du` before recognizing an already exact node-local copy. Do not mutate the image's system Python packages solely to repair `gsutil`.
- Observed: Cloud SDK `428.0.0` could list, read, and copy the regional OpenWebText objects but did not expose `gcloud storage rsync`; its `gsutil` failed during the OpenSSL import. A `gcloud storage cp` fan-out produced the canonical `78` objects totaling `37,384,191,384` bytes on each worker.

### Composite checkpoint download requires compiled CRC32C

- Symptom: `gsutil cp` can read object metadata but refuses a composite checkpoint with `Downloading this composite object requires integrity checking with CRC32c` because `crcmod` is not using its C extension.
- Cause: composite GCS objects expose CRC32C rather than MD5, while the legacy VM image has only slow pure-Python CRC32C support and gsutil fails closed for a large download.
- Recovery: first verify the object, regional bucket, expected bytes, and a trusted SHA-256 from its completion marker. Download to a fresh run-scoped temporary path with `gsutil -o "GSUtil:check_hashes=never" cp`, then independently require exact bytes and SHA-256 before setting the file read-only and atomically renaming it to `checkpoint_N`. Never promote an unchecked file, and never use this fallback without a trusted strong hash.
- Prevention: preflight whether the object is composite and whether compiled `crcmod` or a working `gcloud storage` client is available. Prefer the normal client integrity path; retain the independently verified SHA-256 gate in either case.
- Observed: layer-mixing checkpoint `364800` was a 32-component object. Normal gsutil exited before leaving a partial file; the guarded fallback produced exactly `1,674,052,561` bytes with SHA-256 `ddcedc35f9aa8fac0e99ace7761d6116a15ba8601624d13fa10e9c02276d85ff`.

### Legacy parallel `gsutil` workers cannot load compiled CRC32C

- Symptom: `gsutil version -l` reports `compiled crcmod: True`, including with
  user-site packages disabled, but `gsutil -m rsync` subprocesses reject every
  composite dataset shard as though the C extension were unavailable.
- Cause: the legacy parallel worker launcher resolves a different bundled
  Python/module path from the preflighted parent process. Installing `crcmod`
  into user and system Python therefore does not repair the `-m` children.
- Recovery: preserve the failed zero-step attempt and confirm there is no live
  transfer. Test one shard with serial `gsutil cp` under the exact system Cloud
  SDK Python. If its normal CRC check succeeds, run serial `gsutil rsync -r`,
  then require exact source/local aggregate bytes, all required metadata and
  shards, and zero `.gstmp` files before launch. Never disable hash checks merely
  to retain parallelism.
- Prevention: test a representative composite shard, not only `gsutil version
  -l`, before a first large dataset sync on an old image. Prefer a working
  `gcloud storage` client or a verified node-local dataset when available.
- Observed: PG19 retry24 on a 2022-era v4 image rejected all 23 train shards
  through `gsutil -m` despite compiled `crcmod 1.7` parent preflights. Serial
  CRC-checked sync completed all 46 Arrow shards and produced an exact
  37,799,932,705-byte local snapshot with its manifest and no temporary files.

### Cross-region or non-local training path

- Symptom: `data_path`, source, cache, output, or checkpoint resolves to NFS/Filestore or a mismatched GCS region.
- Cause: reused host path, default bucket fallback, or incomplete override.
- Recovery: refuse launch. Stage once under `/home/$USER/elf-data` and `/home/$USER/elf-runs`; verify bucket location equals TPU region before export.

## Runtime dependency and import failures

### Launcher and initializer disagree on the pinned-runtime profile label

- Symptom: the runtime initializer verifies the expected pinned packages, but
  the staging launcher exits locally with `unsupported ELF_RUNTIME_PROFILE`
  before source staging.
- Cause: the initializer identifies its fixed environment by concrete package
  versions while the staging launcher may expose a different symbolic profile
  label for that same environment. Passing a package-version-style name that
  only one side recognizes fails even when the installed versions are exact.
- Recovery: preserve the zero-step attempt under its original identity. Select
  a fresh output and W&B identity, use the profile label accepted by the exact
  staging launcher, repeat availability and occupancy gates, and relaunch from
  a new immutable snapshot. Do not reinstall an already verified runtime merely
  to repair a label mismatch.
- Prevention: keep one canonical profile-name mapping shared by initialization
  and staging, or add a focused contract test for every supported alias. Record
  both the symbolic profile and resolved package versions in launch evidence.
- Observed: the grey checkpoint 304272 embedding sweep v1 passed JAX 0.6.2
  runtime initialization, then the outer launcher rejected profile
  `jax-0.6.2`; that launcher calls the same pinned environment `default`. The
  command exited locally before cloud access, source copy, W&B, or step 1.

### Later dependency install drifts an earlier exact runtime pin

- Symptom: initialization writes `READY` and all imports succeed, but the
  launcher's exact all-worker version gate reports the runtime missing or
  incompatible on every worker.
- Cause: a later resolver-backed installation changes a package pinned in an
  earlier installation command. Import-only validation then accepts the drift
  and writes a misleading readiness marker.
- Recovery: preserve the zero-step attempt and use a fresh retry identity.
  Move or repeat exact pin enforcement after every dependency-resolving install,
  validate concrete versions CPU-only, repair every idle selected worker, and
  require the launcher's unchanged exact gate to pass before staging source.
- Prevention: make exact version assertions—not mere imports—the final package
  action before `READY`; add a focused ordering/contract test that the last
  resolver pass is followed by pin enforcement and exact validation.
- Observed: the grey checkpoint embedding sweep v2 found Orbax Checkpoint
  0.11.39 on all workers after the initializer had requested 0.11.25 before its
  later `mauve-text` install. JAX/JAXlib 0.6.2 and the other inspected pins were
  correct, but the launcher stopped before source staging or step 1.

### CPU-only test collection imports Torch-XLA and segfaults

- Symptom: `JAX_PLATFORMS=cpu pytest ...` exits `139` during collection while `train.py` imports Transformers; the fatal stack enters `accelerate.utils.imports` and `torch_xla/__init__.py` before any test executes.
- Cause: Transformers/Accelerate discovers the installed optional Torch-XLA package even though the control-host test is explicitly CPU-only. An installed TPU extension is not necessarily safe to import in that control-host process.
- Recovery: preserve the failed command and package versions, then rerun the same CPU-only tests with `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu`. Treat a clean rerun as test-environment recovery, not as evidence that a prior assertion failed.
- Prevention: add `USE_TORCH_XLA=0` to control-host test commands that import `train.py`, Transformers, or Accelerate. Do not apply it to the TPU VM training process.
- Observed: the self-conditioning-feedback prelaunch suite crashed with Torch `2.7.0+cpu`, Torch-XLA `2.7.0`, Transformers `5.2.0`, and Accelerate `1.12.0`; the identical suite then passed `21` tests after disabling optional Torch-XLA discovery.

### W&B credential file is group/world accessible

- Symptom: `initialize_elf_runtime.sh` exits before cloud access with `ELF W&B credential file must have mode 0600 or 0400`.
- Cause: the local `.netrc` contains a W&B credential but has permissions that allow other users to read or modify it.
- Recovery: record the zero-step initializer attempt, change only that credential file to owner-readable/writable mode `0600`, revalidate its ownership and mode without printing its contents, and rerun the idempotent initializer.
- Prevention: require mode `0600` or `0400` before every credential copy and never stage `.netrc` inside a source snapshot.

### Isolated launch worktree has no credential file

- Symptom: configuration resolution and initial TPU connectivity succeed, but the runtime initializer exits with `ELF W&B credential file is missing: <isolated-worktree>/.netrc` before source or data staging.
- Cause: the initializer resolves its default `.netrc` relative to the selected source worktree, while credentials are intentionally excluded from clean worktrees and immutable snapshots.
- Recovery: preserve the zero-step attempt, use a fresh output/W&B identity, verify an owner-only credential outside the isolated worktree without printing it, and pass that exact path through `ELF_NETRC_PATH` to the outer initializer. Keep `.netrc` excluded from every source snapshot.
- Prevention: when `ELF_STAGING_SOURCE` or a separate launch worktree is used, preflight the effective credential path independently of the source root and require owner plus mode `0600` or `0400` before any cloud mutation.

### Venv executable exists but initialization is incomplete

- Symptom: preflight finds `venv/bin/python`, but required imports such as `datasets` and `wandb` fail and W&B authentication cannot be checked.
- Cause: an asynchronous initializer was interrupted after creating the venv but before installing and validating the pinned dependency set; executable existence alone is not a readiness marker.
- Recovery: reject the launch at preflight, confirm no initializer or accelerator holder is active, rerun the idempotent runtime initializer, and require its final `READY` marker plus complete CPU import and W&B-authentication checks before assigning a fresh retry identity.

### Manual checkpoint probe uses a stale runtime path

- Symptom: a checkpoint reaches exact byte/SHA verification and read-only atomic promotion, but its CPU payload probe exits `127` because the referenced virtual-environment Python path does not exist.
- Cause: the manual probe hard-codes a legacy runtime directory instead of using the selected launcher's effective `runtime_python` path.
- Recovery: preserve the promoted read-only checkpoint and the zero-step attempt. Resolve the runtime path from the exact staged launcher, then re-run only size, SHA-256, and CPU `msgpack_restore` checks. Never redownload or rewrite a checkpoint whose marker-backed bytes already passed.
- Prevention: source one runtime-path contract for staging, payload inspection, import checks, and launch; preflight `test -x` before transferring a multi-gigabyte checkpoint.

### System Python is too old for the pinned ELF runtime

- Symptom: initialization fails its Python version assertion before the venv can install packages; observed on an Ubuntu 20.04 v4 image with Python 3.8.10.
- Cause: the TPU image predates the Python version required by the pinned ELF dependency set.
- Recovery: bootstrap a pinned node-local Python with pinned `uv`, seed pip explicitly, and leave the image's system Python unchanged. Verify the complete import set on CPU before any TPU probe.

### Cloud SDK lacks GA `gcloud storage`

- Symptom: `gcloud storage` reports `Invalid choice: 'storage'`; legacy alpha storage may also lack flags such as `du --summarize`.
- Cause: an older v4 image has a Cloud SDK predating the GA storage surface.
- Recovery: use `gcloud storage` when supported and otherwise use installed `gsutil` equivalents for `du`, `rsync`, `cp`, and `rm`; verify exact bytes and dataset metadata. If `gsutil` has the OpenSSL failure documented above, reject the image rather than using incompatible alpha syntax.

### Missing `sacrebleu`

- Symptom: `ModuleNotFoundError: No module named 'sacrebleu'` while importing `generation` through `utils.metrics_utils`.
- Cause: fresh TPU image lacks an eagerly imported evaluation dependency even when online evaluation will not run immediately.
- Observed repair: `python3 -m pip install --user sacrebleu==2.6.0 rouge-score==0.1.2 einops==0.8.2`.
- Verify before staging large data when possible: `JAX_PLATFORMS=cpu PYTHONPATH=<source> python3 -c 'from generation import run_generation'`.
- Record: old/new package versions, full-import result, failure at step 0, and absence of TPU holder/checkpoint.

### Missing `rouge_score`, `einops`, or transitive imports

- Symptom: import error advances to the next eagerly imported module after one package is repaired.
- Cause: requirements do not fully encode runtime imports or the image has an incomplete environment.
- Recovery: use versions validated by prior ELF runs; rerun the full top-level import, not only the previously missing package import.

### Minimal TPU runtime omits the pytest test runner

- Symptom: an explicitly staged kernel validation exits with `No module named pytest` before importing the test or initializing the TPU backend.
- Cause: the node-local ELF runtime contains production dependencies but not the development-only pytest runner.
- Recovery: preserve the zero-step attempt and do not mutate the shared runtime solely for one validation. If the staged test module has a guarded `__main__` entry point, execute it directly with the same runtime Python and explicit test environment; otherwise use a distinct validated test environment and record its exact pytest version.
- Verify: require the direct module or replacement runner to report a TPU backend and the complete intended parity result; a successful import alone is insufficient.

### Optax lacks Muon

- Symptom: `AttributeError: module 'optax.contrib' has no attribute 'muon'` during optimizer construction.
- Confirmed cause: `optax==0.2.1` lacks the required API.
- Observed repair: `python3 -m pip install --user optax==0.2.8` while retaining JAX/JAXlib 0.6.2.
- Verify: print JAX/Optax versions, assert `hasattr(optax.contrib, "muon")`, confirm TPU device count, and confirm the device is idle before retry.
- Record as a separate zero-step attempt even if model construction succeeded.

### JAX/JAXlib/libtpu incompatibility

- Symptoms: backend initialization error, plugin ABI/version error, no TPU devices, or CPU fallback.
- Causes: pip upgraded only part of the JAX stack, incompatible TPU wheel, stock startup installer clobbered `~/.local`, or wrong interpreter/PYTHONPATH.
- Prevent: wait for image startup package work; pin the validated environment; require `JAX_PLATFORMS=tpu` and positive TPU device count after all installs.
- Recovery: do not launch until the final interpreter resolves compatible JAX, JAXlib, Optax, Flax, and libtpu.

### W&B unavailable on one or more workers

- Symptoms: missing API key on physical worker assigned process 0; other workers time out during distributed shutdown.
- Cause: credentials verified only on worker 0 even though process 0 placement is not fixed.
- Recovery: require authentication on every worker or disable W&B explicitly. Never print credentials.

### W&B helper opens a conflicting libtpu client

- Symptoms: W&B initializes, a second timestamped libtpu process appears on the tracking worker, its `tokio-rt-worker` may segfault, and the first distributed program later halts with an unexpected peer or different launch ID.
- Cause: W&B hardware/system-stat discovery inherited `JAX_PLATFORMS=tpu` and opened the accelerator independently of the intended trainer.
- Recovery: initialize W&B on launcher-identified worker zero before `jax.distributed.initialize()`; temporarily force W&B subprocesses to `JAX_PLATFORMS=cpu`, disable W&B system-stat collection, restore the parent trainer's TPU platform, and retain ELF application metric logging. Verify exactly one intended libtpu client per worker before attributing any remaining launch-ID mismatch to W&B.
- Observed: retry7 opened helper libtpu process 110022 before intended trainer process 109834. Retry8's CPU isolation and disabled stats left only intended process 117284 and no helper kernel fault.

### Hugging Face/model artifact failure

- Symptoms: authentication error, `404`, checksum failure, timeout, or missing encoder checkpoint/tokenizer.
- Causes: gated/private repo, missing token, wrong revision, unavailable network, or incomplete cache.
- Recovery: verify artifact names/revisions and node-local cache on every worker; record credential presence only, never token values.

## TPU backend and distributed initialization failures

### Stale `/tmp/libtpu_lockfile`

- Symptom: TPU backend says the device is already in use while `lsof /dev/vfio/0` shows no live holder; stale lockfile remains from a terminated workload.
- Cause: previous process ended without clearing libtpu state.
- Recovery: do not reboot. Record lock ownership, UID, mode, size, timestamp, and the absence of both a lockfile and VFIO holder on every worker. Use a separately safe authorized recovery path or another TPU; do not delete or move another user's lock without explicit authority. When the user explicitly authorized reclaiming that exact foreign job, preserve a revalidated zero-byte stale lock under a unique timestamped quarantine name instead of deleting it, then require the original path to be absent, the quarantine artifact to retain its owner/size, and every accelerator resource to remain idle before a fresh retry identity.
- Prevention: both the initial and final launch preflights must reject the mere existence of an unheld `/tmp/libtpu_lockfile` as `stale_libtpu_lock`; checking only `fuser` is insufficient because libtpu refuses initialization even when the file has no live holder.
- Observed: the second v6e-32 fused linear-mask gate passed VFIO checks but failed on all eight workers before step 1. Every worker retained the same unheld zero-byte mode-600 lock owned by Zak (UID 2019) from `2026-08-06T12:13:21Z`; the corrected exact-target guard log was empty. The locks were preserved as `/tmp/libtpu_lockfile.stale-zak-20260806T121321Z` only after the user's exact Zak-job reclaim authorization and complete all-worker revalidation.
- Observed again: the v5e-16 local-four XLA MFU baseline's standard inventory helper reported all four workers `AVAILABLE`, but launch preflight then rejected unheld current-user locks on three workers before source or data staging. The helper had counted only lock holders, not the canonical lock path itself. It now reports a holderless lock as protected `OCCUPIED` with metadata in both inspection and claim modes.
- Observed after successful completion: the v5e-32 JAX 0.11 gradient-parity
  diagnostic reached its configured step 20 and exited cleanly, but every
  worker retained a current-user, zero-byte, unheld canonical lock. After
  confirming the terminal sentinel, no matching trainer, and idle VFIO on all
  workers, the eight locks were preserved under the unique
  `libtpu_lockfile.completed-gradient-parity-jax011-retry1-20260818T1750Z`
  quarantine name. A successful training exit is not proof that libtpu removed
  its lockfile.

### Collaborator-owned `/tmp/tpu_logs` blocks libtpu startup

- Symptoms: every worker repeatedly emits `Could not open the log file ... Permission denied`, followed by the 60-second multi-host initialization warning; no first step or W&B run appears.
- Cause: `/tmp/tpu_logs` persists from a collaborator workload as a non-world-writable directory such as mode `0775` with that collaborator's ownership.
- Recovery: stop only the exact zero-step launch, preserve its logs, verify every VFIO device is idle, and record the directory owner/mode. With explicit authority to reclaim that collaborator's workload state, normalize only `/tmp/tpu_logs` to `root:root` mode `1777`, clear the stopped launch's own stale libtpu lock, rerun preflight, and relaunch the same immutable configuration. Never touch checkpoints or user data.

### Different-incarnation/coordinator abort

- Symptoms: distributed coordinator rejects a task with a different incarnation, then workers receive fatal aborts.
- Cause: a second multihost JAX workload connected to the same TPU while ELF was running.
- Evidence: new foreign SSH/process activity, no reboot/OOM/preemption, and concurrent device ownership.
- Recovery: let processes exit naturally, preserve checkpoints, and resume only after every worker is idle. Never share a TPU.

### Single-client multi-host launch-ID divergence

- Symptoms: libtpu reports `An unexpected peer shows up in the launch group with a different launch id`, but every worker has exactly one intended trainer client and no W&B/helper process opened libtpu.
- Cause: the duplicate-local-client class is rejected, but the positive root
  cause remains unconfirmed without stronger evidence. Candidates include the
  JAX/JAXlib/libtpu runtime build, compiler optimization/lowering, and
  nondeterministic host/model construction across controllers. Do not label
  model code or XLA compilation as confirmed merely because only one client is
  present.
- Recovery: preserve the zero-step evidence. First compare the exact scientific
  configuration under a byte-identical, previously validated runtime on every
  worker and a fresh run identity. If the failure remains, use before/after HLO
  dumps from every worker when root-level compiler diagnosis is required, or
  move the unchanged scientific experiment to a proven single-host topology.
  Preserve per-chip batch by scaling global microbatch with device count,
  preserve effective optimizer batch with accumulation, and rerun HBM preflight
  under a distinct identity.
- Backend-independence criterion: if the same multi-host launch-ID divergence reproduces under both fused Pallas and dense XLA attention, reject the attention backend as the sole cause. This implicates host-side model/program construction or compiler nondeterminism across controllers; identifying the exact divergent component requires comparing per-worker HLO/compiler artifacts. For production recovery, prefer a scientifically equivalent single-host topology instead of cycling attention backends on the failing multi-host topology.
- Observed: retry8 had only process 117284 on worker zero and no kernel fault, yet its four-host v6e-16 program diverged. Retry9 moved from global 256/accumulation 2 on 16 chips to global 128/accumulation 4 on a single-host 8-chip v6e, preserving 16 examples/chip and effective batch 512; it passed compilation and reached step 1,900.
- Observed again: the stale-lock-corrected v6e-32 fused linear-mask gate reached distributed initialization, constructed the 104,584,548-parameter model across eight hosts and 32 devices, and then halted during its initial compiled step with the exact different-launch-ID on-device check. W&B was disabled, and each worker produced exactly one intended libtpu INFO client log and one ERROR log, ruling out duplicate local clients before selecting the proven single-host v6e-8 recovery.
- Observed with dense XLA: the otherwise unchanged v6e-32 fallback gate constructed the same 104,584,548-parameter model and Muon optimizer on all eight hosts, then reproduced the different-launch-ID on-device failure before step 1. The cross-backend reproduction blocked full v6e-32 promotion and selected single-host production recovery.
- Observed with matched pre-HLO and divergent post-HLO: a four-host v4-32 dense-XLA Gate7 emitted exactly one `pmap_train_step` before/after HLO pair per worker. All four `11,431,082`-byte before-optimization files shared SHA-256 `3f12aa87903484ca76b9c2b2825c0c8d41b9c7070508958d0bb8aecd17dea84d`, while every optimized file had a distinct size, line count, SHA-256, and copy/custom-call instruction count. This directly rejected host/model tracing divergence for the captured step and localized the first proven divergence to TPU compiler optimization/lowering; the exact compiler pass remained unconfirmed. For a subsequent multi-controller promotion, require step 1 plus exactly one before/after train-step HLO per worker and byte-consistent optimized HLO, or test a fresh isolated TPU runtime before falling back to a scientifically equivalent single-host topology.
- Observed with a runtime-controlled recovery: an eight-host v5e-32 exact 32/32
  same-gradient diagnostic used identical Python 3.10, JAX/JAXlib 0.6.2, and
  libtpu hashes on every worker but failed before step 1 with the launch-ID
  check. After exact orphan cleanup, all workers received an identical Python
  3.12, JAX/JAXlib 0.11.0 runtime; a fresh immutable retry of the same scientific
  workload completed 20/20 steps. This makes the pinned 0.11.0 runtime the
  operational requirement for that path, but it does not prove that version
  alone explains every launch-ID failure.

### Mixed ELF runtime signatures across TPU workers

- Symptoms: all workers have one trainer and healthy/idle TPU devices, but the first compiled multi-host step fails with the different-launch-ID check; per-worker Python minor versions or `libtpu.so` SHA-256 values differ.
- Cause: the initializer selected each VM image's ambiguous `python3` default. In a mixed-image slice that installed distinct Python ABI wheels and therefore distinct libtpu binaries despite the same nominal JAX version.
- Recovery: preserve the failed output path, repair only the divergent unused node-local venv using the shared pinned Python version, verify the complete runtime signature on every worker without initializing TPU, and relaunch with a fresh run identity. Preserve the previous venv until the repair succeeds.
- Prevention: pin the bootstrap Python interpreter, and immediately before every multi-host launch require one shared Python-major/minor, JAX, JAXLIB, and libtpu SHA-256 signature; fail closed on a mismatch.
- Observed: PG19 40x classic fused-Pallas v6e-32 retry7 had Python 3.10/libtpu `55e1b223…ece17f8` on workers 0--6 but Python 3.11/libtpu `4c92530d…a5d6ea` on worker 7. Rebuilding worker 7 with managed Python 3.10 produced the shared hash before fresh retry8.

### Identical runtime signatures fail comparison after SSH fan-out

- Symptom: the prelaunch runtime gate prints the same Python, JAX, JAXLIB, and libtpu digest for every worker, but still reports a runtime mismatch.
- Cause: SSH fan-out can preserve transport carriage returns or trailing whitespace, so visually identical signature records remain distinct byte strings under `sort -u`.
- Recovery: preserve the zero-step attempt, normalize carriage returns and trailing whitespace only, then rerun the gate under a fresh identity. Do not weaken or skip the payload comparison.
- Prevention: canonicalize transport line endings and trailing whitespace before comparing runtime signature payloads; retain an exact regression test with both CRLF and space-padded duplicates.
- Observed: the v5e-16 local-four XLA MFU baseline retry 2 initialized one pinned runtime on every worker and printed four identical libtpu digests, but the unnormalized gate rejected them after dataset localization and before output claim or TPU backend initialization.

### Host OpenSSH configuration is malformed or changes between attempts

- Symptoms: otherwise valid `gcloud ... tpu-vm ssh` or SCP commands intermittently fail with `/etc/ssh/ssh_config line 54: no argument after keyword` followed by varying non-printable keyword bytes.
- Cause: the host OpenSSH configuration is corrupted or being concurrently rewritten; retrying without isolation can produce nondeterministic launch and monitoring failures unrelated to TPU health.
- Recovery: forward `-F` and `/dev/null` as separate arguments on every gcloud SSH path (`--ssh-flag=-F --ssh-flag=/dev/null`) and every SCP path (`--scp-flag=-F --scp-flag=/dev/null`). Apply the same isolation to discovery occupancy probes, final claim checks, launch staging, and recurring monitoring, then require focused command-construction tests.
- Observed: the staging, watcher, filtered inventory, and guarded occupancy helper all required this isolation during the 2026-08-06 recovery.

### Concurrent snap-packaged gcloud commands cannot create transient scopes

- Symptom: several otherwise valid TPU staging commands pause in unrelated `gcloud` phases and then fail together with `internal error, please report: running "google-cloud-cli.gcloud" failed: transient scope not created in 10s`; their transactional rollback SSH commands may stall for the same reason.
- Cause: the host's snap-packaged Google Cloud CLI cannot create all requested transient execution scopes under the concurrent launch burst. This is a control-host packaging failure, not evidence of TPU, runtime, dataset, checkpoint, or W&B failure.
- Recovery: preserve every zero-step attempt under a distinct report and retry identity. Close only the exact failed local staging processes and their proven children, then verify every target worker has no output, source-claim marker, trainer, VFIO holder, or libtpu lock. Retry staging serially on the control host. If the serialized command repeats the transient-scope error and `/snap/bin/gcloud` resolves through `/usr/bin/snap`, invoke the same authenticated CLI through `/snap/google-cloud-cli/current/bin/gcloud` (for example, an exported shell function named `gcloud`) and record this control-plane-only entrypoint substitution. Once each foreground launch reaches its TPU-side trainer, the evaluation workloads may continue in parallel.
- Prevention: stagger immutable staging and launch commands, retain each required foreground launch session, and do not begin the next staging command until the previous one has crossed the final VFIO gate and started its trainer. Prefer the direct packaged `gcloud` binary when the Snap alias has already reproduced the transient-scope failure serially. Before retry, retire foreground CLI/SSH trees only for runs independently proven terminal and archived.
- Observed: four FineWeb classic checkpoint evaluation preflights for steps 800k, 2.0M, 2.4M, and 2.6M passed configuration and runtime checks, then failed together during a four-command control-host launch burst. Transactional verification found every output/source claim absent, every trainer absent, and all selected VFIO resources idle before distinct staggered retries.

### Gcloud all-worker TPU SSH reconnect returns a malformed identification banner

- Symptom: a TPU VM address fails before authentication with `kex_exchange_identification: banner line contains invalid characters`, even with `-F /dev/null`; the public key is already present and a prior all-worker SSH fan-out may have completed successfully.
- Cause: the selected port-22 path is returning data that is not a valid SSH identification stream during the gcloud all-worker reconnect path. The lower-level transport trigger remains unconfirmed, but the failure is localized away from authentication: the TPU sshd logs show the internal control host resetting before authentication, no MaxStartups throttling, and successful public-key authentication from the same host on explicit single-worker calls. Do not attribute the failure to `authorized_keys`, `ssh-agent`, or a missing shell variable merely because the wrapper prints its generic `ssh-add` suggestion.
- Recovery: confirm the exact authorized `kristine` TPU is `READY` and `HEALTHY`, use `ELF_TPU_USE_INTERNAL_IP=1`, and replace post-install `--worker=all` reconnects with explicit worker-index SSH/SCP calls. Retry each idempotent worker phase with bounded backoff and require every worker's READY marker, CPU imports, Muon API, and W&B authentication before accepting initialization. Never copy a private key to the TPU, modify duplicate public-key entries, or reboot as a transport repair.
- Prevention: keep only the long dependency installation in a parallel all-worker fan-out when validated. Run readiness, credential transfer, and finalization worker by worker; keep these phases idempotent and tested. A successful package-install fan-out does not prove the next gcloud all-worker reconnect will work.
- Observed: the us-west4-a v5e-32 TPU `kmh-tpuvm-v5e-32-spot-kristine-10-17-33-08-N-us-west4-a` was `READY` and `HEALTHY`. The all-worker reconnect failed on external and internal addresses after dependency installation. Every worker had one active sshd listener with `Banner none`; worker 0 had zero MaxStartups events and recorded the control host resetting pre-authentication. The repaired initializer then connected to workers 0 through 7 individually, verified every runtime, and exited `0` with `RUNTIME_INITIALIZATION=REUSED`.

### One TPU worker closes SSH after accepting the public key

- Symptom: some workers authenticate, but one worker consistently ends with `Connection closed by <address> port 22`; verbose SSH reaches key exchange, reports that the server accepts the offered key, sends the signed authentication request, and is then disconnected. Guest attributes can still report the VM as `sshable` because that attribute proves listener readiness rather than a complete user login.
- Cause: the failing worker's guest authentication/account path is unusable. When the same failure reproduces with RSA and Ed25519 keys, internal and external addresses, and direct OpenSSH outside gcloud, it is not caused by a missing metadata key, `ssh-agent`, the gcloud retry wrapper, or an unset ELF environment variable. The worker-side PAM/account reason can remain unconfirmed when every authenticated session is closed before a command can inspect daemon logs.
- Recovery: fail closed before dependency setup or credential transfer and reject the entire multi-worker slice. Do not initialize the reachable subset, copy a private key onto a TPU, duplicate the accepted public key, or reboot/reimage as an SSH workaround. Use a different authorized `kristine` TPU unless a separate explicitly authorized infrastructure workflow can repair or recreate the defective worker.
- Prevention: resolve the selected internal or external address for every worker and require a bounded direct-SSH, public-key-only `true` command to succeed on all workers before any mutation. Print the exact worker and address on failure. After the gate, use direct per-worker SSH/SCP calls and preserve parallelism for long installation work without returning to an unbounded gcloud all-worker reconnect.
- Observed: the `READY`/`HEALTHY` us-west4-a v5e-32 TPU `kmh-tpuvm-v5e-32-spot-kristine-11-12-46-15-N-us-west4-a` authenticated workers 0 and 1, then closed worker 2 at `10.182.0.245` after accepting both tested key types. The bounded initializer exited `78` before runtime installation and left the TPU unchanged.

### Distributed worker timeout

- Symptoms: some workers wait at initialization/shutdown barrier and eventually time out.
- Causes: one worker had the original import/credential/backend failure, inconsistent files/environment, unreachable coordinator, or a preempted worker.
- Recovery: find the first error on the earliest worker; treat later timeouts as secondary effects.

### Hard-coded process-count mismatch

- Symptoms: worker count/incarnation errors or hang during `jax.distributed.initialize()`.
- Cause: hard-coded `JAX_PROCESS_COUNT`, assumed eight workers, or inconsistent fan-out.
- Recovery: remove hard-coded process counts and let Cloud TPU/JAX discover workers; stage and launch with `--worker=all`.
- Observed: the first official ELF-B/M/L distribution collector attempt started
  only physical worker 0 on an eight-host v4-64 slice while its code required
  `jax.process_count() == 1`. TPU backend initialization waited for the seven
  absent hosts and emitted the 60-second all-host warning before any checkpoint
  restore or sequence. The retry used automatic distributed initialization on
  all eight workers, a disjoint distributed held-out sampler, and a global
  sequence-count gate.

### Pre-distributed logging initializes the JAX backend

- Symptom: every worker exits at `jax.distributed.initialize()` with `must be called before any JAX calls that might initialise the XLA backend` immediately after launcher config overrides are applied.
- Cause: the pre-initialization override message used `log_for_0()`, which calls `jax.process_index()`. That process query initializes the backend before the explicit multi-host coordinator is formed.
- Recovery: preserve the zero-step attempt and use a fresh output/W&B identity. Log pre-initialization launcher messages only from the launcher-provided worker index, without any JAX API; call `jax.distributed.initialize()` before the first `jax.process_index()`, `jax.devices()`, array construction, or device transfer.
- Prevention: exercise the versioned-run override path in a focused entrypoint test that makes any pre-start `log_for_0()` call fatal. Parse `wandb_run_id=None` as a real null so retry identity overrides do not create the unrelated literal W&B run ID `None`.
- Observed: FineWeb v5e-32 retry `20260811-elf-b-fineweb-edu-10bt-512-512-classic-xla-v5e-32_v2` failed identically on worker 0 and worker 1 before distributed initialization or step 1. The TPU remained `READY`/`HEALTHY`, every trainer exited, no checkpoint existed, and worker 0's W&B setup emitted the literal `runs/None` URL from the older snapshot's override parser.

### Gcloud endpoint formatter collapses a multihost slice onto one line

- Symptom: gcloud SSH identifies multiple workers, but launch preflight exits
  with `explicit_multihost_jax_topology requires at least two TPU workers; found
  1`.
- Cause: `--format='value(networkEndpoints[].ipAddress)'` serializes the
  repeated endpoint field as one semicolon-delimited line. A launcher that uses
  `mapfile` and counts lines mistakes that line for one endpoint.
- Recovery: preserve the zero-step attempt and parse the repeated values from
  structured JSON or explicitly split gcloud's list delimiter. Require every
  parsed value to be a valid IPv4 address before constructing coordinator and
  process-count variables.
- Prevention: add a focused command-construction/parser test whose endpoint
  output matches gcloud's real semicolon-delimited representation; do not mock
  repeated fields as newline-separated when production does not emit them that
  way.
- Observed: the `READY`/`HEALTHY` us-west4-a `v5litepod-32` FineWeb TPU exposed
  eight API endpoint objects, while gcloud formatted their addresses on one
  line. The launcher counted one line and stopped before training. Normalizing
  semicolons to newlines restored the correct count while retaining the IPv4
  validation; the complete staging suite passed afterward.

### Topology exports escape the remote launch-script writer

- Symptom: launch reaches `STATUS=LAUNCHING`, prints a bare `export` and a
  coordinator assignment, then Bash reports that `ulimit -c 0` and the `exec
  ... train.py` command are invalid identifiers supplied to `export`.
- Cause: a variable containing multiple semicolon-separated export commands is
  interpolated into a `printf` command inside an already double-quoted remote
  command string. The interpolation is not preserved as one quoted `printf`
  argument, so the first semicolon ends `printf` and the last `export` consumes
  the following intended script lines as arguments.
- Recovery: preserve the all-worker step-zero attempt. Render the coordinator,
  process count, and process index as separately quoted lines in the generated
  script, then syntax-check the fully rendered script before any foreground
  fanout. Use a fresh output identity for the retry.
- Prevention: test the generated launch script itself, including `bash -n`, one
  topology assignment per line, and the final `ulimit` and `exec` lines. A test
  that only searches the outer gcloud command cannot detect nested quote loss.
- Observed: the FineWeb v5e-32 timestamped retry transmitted the malformed
  command to all eight workers, and every worker exited `1` before starting a
  trainer.

## Compilation and first-step failures

### TPU out of memory

- Symptoms: XLA resource exhaustion/OOM during compile or first step.
- Causes: topology below the required 2x HBM margin, excessive per-chip batch, trainable encoder, optimizer/EMA state, activations, or compiler buffers.
- Recovery: record memory assumptions and exact failing configuration. Reduce per-chip microbatch while preserving effective batch with accumulation, or request a larger authorized topology; never silently change model semantics.
- Observed: ELF-B batch `256` on v6e-8 required `42.02 GiB` per chip versus `31.25 GiB`; the two largest allocations were separate `3.89 GiB` FP32 decoder vocabulary tensors shaped `[32, 1012, 32100]`. Exact chunked linear cross-entropy is the semantics-preserving repair when full-sequence logits dominate HBM.

### Pallas custom call exceeds scoped VMEM

- Symptom: model construction or compilation fails with `RESOURCE_EXHAUSTED` and reports a scoped VMEM request above the TPU's per-program limit even though aggregate HBM preflight passed.
- Cause: one Pallas program's input tiles, output tiles, scratch buffers, and compiler-managed buffering exceed VMEM; packed block specs can load unused tensor components and make an otherwise viable tile fail.
- Recovery: preserve the zero-step attempt, record requested and allowed VMEM, and use a distinct retry that reduces the program tile or removes unused packed inputs/scratch. Do not treat additional TPU chips or aggregate HBM as a repair for a per-program VMEM limit.
- Prevention: inspect every large block spec for unused components, test the exact tile through model construction, and retain a smaller verified tile as the fallback.
- Observed: BQ1024 fused ELF attention requested `33.21 MiB` against the v6e `32.00 MiB` scoped limit because the full-sequence query input tile included packed Q, K, and V while the query path used only Q.

### Pallas block uses an illegal TPU minor-dimension layout

- Symptom: TPU lowering rejects a `BlockSpec` because its last two block dimensions are not divisible by `8` and `128`, respectively, and are not equal to the corresponding full-array dimensions; CPU interpret-mode tests may still pass.
- Cause: slicing or transposing an input moved a small logical axis, such as attention heads, into one of Mosaic TPU's two tiled minor dimensions and then attempted to block that axis below its full size.
- Recovery: preserve the zero-step attempt and use a distinct retry that transposes the input so tile-friendly axes occupy the two minor dimensions. For attention components, `[batch, head, sequence, dimension]` permits blocks shaped `[1, 1, block_sequence, head_dimension]` without restoring unused packed components.
- Prevention: add a TPU compile/initialization gate for block-layout changes because Pallas interpret mode does not enforce every TPU lowering constraint.
- Observed: the first split-input BQ1024 ELF retry passed CPU parity but TPU lowering rejected a `[1, 1024, 1, 256]` block over `[1, 1024, 3, 256]`; the one-head penultimate block was neither divisible by `8` nor the full three-head dimension.
- Observed: throughput trial 22 passed compact-statistic CPU forward/VJP parity, then TPU lowering rejected a `(1, 1, 1024)` block over `(32, 3, 1024)` for backward argument 7. The penultimate block dimension `1` was neither divisible by `8` nor equal to the full head dimension `3`; the run failed before step 1 and retained the replicated-statistic winner as fallback.

### Compilation appears stuck

- Symptom: process/device holder exists but no first-step output for an extended period.
- Causes: legitimate XLA compile, deadlock, memory pressure, or kernel/runtime issue.
- Verify: process state, log timestamp, CPU/memory, TPU health, and absence of fatal patterns.
- Recovery: do not report launch success and do not reboot. Continue diagnosis; if kernel-stuck, leave the TPU untouched and move only through an authorized replacement path.

### Host-side quadratic-mask throughput regression

- Symptoms: training is scientifically valid but steady-state examples/second falls materially below the accepted like-for-like benchmark; host CPU, memory, or input-wait time may be high even when a later stage drops the mask.
- Cause: the collator, preparation, or prefetch path materializes or transports a float32 `[batch, sequence, sequence]` attention mask instead of carrying linear `attention_mask` and `cond_seq_mask` values and reconstructing the quadratic mask inside the compiled TPU step.
- Verify: inspect the exact immutable snapshot and host payload keys/shapes. Compute `examples_per_second = (training_loop_steps * global_batch_size) / interval_wall_seconds` over several post-compilation steps, excluding evaluation, profiling, checkpointing, and warmup, then compare the same runtime shape with its recorded benchmark and tolerance. Count `grad_accum_steps` only when the measured interval spans the complete accumulation window; never multiply each loop step by it.
- Recovery: keep a successful smoke as functional/scientific evidence but fail the throughput gate, block full promotion, remove quadratic host materialization, prove linear-payload and on-device-mask parity with a focused fixture, and rerun under a distinct immutable identity.
- Observed: the `20260806` v6e-8 layer-mixing smoke completed at `231.19` examples/s, but its collator built a roughly 1 GiB float32 `[256, 1012, 1012]` mask before the training path discarded it; the full launch remained blocked pending a fixed smoke.

### Invalid config override

- Symptoms: parsing error, wrong type, zero/negative batch, invalid fractional save frequency, missing config field, or `None` interpreted as a string.
- Cause: YAML value type differs from annotated type or launcher validation is incomplete.
- Recovery: add a focused parser/launcher test, preserve the failed attempt, and retry with a distinct output after verification.

### Unknown YAML fields are silently ignored

- Symptoms: a run name or source YAML advertises an architecture/backend setting, but the serialized effective config omits it and runtime behavior uses the model preset or default backend.
- Cause: `load_config_from_yaml` guards assignments with `hasattr(config, key)` and skips undeclared keys without raising, so configuration copied from a newer runtime can appear valid against an older snapshot.
- Recovery: preserve and stop the scientifically invalid attempt, use a distinct output/W&B identity, add the missing typed configuration fields and runtime dispatch, and relaunch only after the fully resolved config contains every requested non-default field.
- Prevention: reject every unknown YAML key during preflight, serialize and inspect the resolved config before staging, and add focused tests proving architecture/backend keys both resolve and reach model construction.
- Observed: the two `20260806` 50k-MAUVE reruns declared `num_attention_heads: 3` and `attention_backend: xla`, but their immutable snapshots lacked both `Config` fields; the loader skipped them and ELF-B remained the hard-coded 12-head XLA preset.

### Derived Hugging Face caches inflate the canonical dataset size

- Symptoms: region-local dataset metadata and every canonical Arrow shard are present with no `*.gstmp` files, but staging enters synchronization and then exits nonzero because the aggregate local byte count exceeds GCS.
- Cause: Hugging Face dataset transforms such as deterministic train/eval splitting write reproducible `cache-*.arrow` files beside the canonical shards. A validator that sums every local file mistakes those derived caches for source-data corruption.
- Recovery: compare the exact canonical object set/bytes against the region-matched GCS prefix while excluding only `cache-*.arrow`; preserve the caches, continue rejecting `*.gstmp`, and keep disk-space checks based on actual free bytes. Do not delete or redownload verified canonical shards.
- Prevention: test that staging's canonical-byte calculation explicitly excludes `cache-*.arrow` while its incomplete-transfer gate still scans the entire dataset tree.
- Observed: the classic Pallas v4 smoke had all 78 canonical objects totaling `37,384,191,384` bytes plus two derived caches totaling exactly `79,300,352` bytes. The old all-file sum was `37,463,491,736`; no canonical object differed and no incomplete transfer existed.

### Shape, sharding, or batch divisibility error

- Symptoms: reshape/sharding exception, per-host batch mismatch, or nondivisible global batch.
- Causes: topology changed without adapting global batch/accumulation or dataset batch semantics.
- Recovery: derive host/device counts at runtime and verify local/global/effective batch before compilation.

### NaN/Inf on first step

- Symptoms: nonfinite loss/gradient or immediate divergence.
- Causes: numeric instability, incompatible precision, corrupt data, extreme learning rate, invalid latent statistics, or bad checkpoint.

## Monitoring failures

### Recurring shell loop ignores systemd TERM during rotation

- Symptom: `systemctl --user stop <monitor>.service` remains active behind its
  stop timeout; the loop's bash and `sleep` children persist or become zombies,
  and later user-manager calls can block or return `Transport endpoint is not
  connected`.
- Cause: a transient unit runs `while true; do <watcher>; sleep N; done`
  without a TERM/INT trap. TERM interrupts a child, but the parent shell
  continues the loop instead of exiting.
- Recovery: revalidate the exact unit PID, process group, command, and children;
  terminate only that host-side monitor group. Confirm no watcher remains and
  never signal the remote trainer. If the user manager remains unavailable,
  avoid restarting it because that can disrupt unrelated services; start one
  detached, PID-tracked recurring loop and require a healthy first check plus a
  single cadence-sleep child.
- Prevention: install recurring wrappers with an explicit TERM/INT exit trap
  and a bounded stop timeout. Test unit rotation before depending on it for a
  long run, and keep the one-shot watcher independent of the wrapper.
- Observed: PG19 retry24's transient systemd monitor failed to stop while adding
  a resume-aware first-evaluation boundary. Exact group cleanup left the TPU
  trainer untouched; the final TERM-safe replacement reached step 405400 and
  slept on the intended 1,800-second cadence.

### Detached monitor exits with its managed sandbox

- Symptom: a `nohup setsid` monitor completes one healthy check and enters its
  cadence sleep, but its PID disappears immediately after the command that
  launched it returns.
- Cause: the launcher itself ran inside a managed process sandbox configured
  with `--die-with-parent`. Shell detachment does not escape that outer process
  lifetime boundary.
- Recovery: confirm the missing PID and preserve the completed check. Launch a
  TERM-safe, PID-tracked wrapper through an approved host execution path, then
  require a second healthy check and exactly one cadence-sleep child. Do not
  touch the remote trainer.
- Prevention: distinguish shell detachment from sandbox lifetime isolation;
  start required persistent monitors outside any `--die-with-parent` boundary
  or through a verified persistent service manager.
- Correction: a later host cgroup audit found PG19 retry24 PID `3087234` still
  alive after a sandbox-scoped process query had reported it absent. That
  namespace-limited query did not confirm a `--die-with-parent` exit; require a
  host-visible PID and cgroup check before classifying this failure.

### Successor monitor inherits an obsolete precursor systemd cgroup

- Symptom: a successor monitor survives the command that launched it and emits
  healthy checks, but `systemctl --user status` for an obsolete precursor unit
  lists that successor monitor, its recovery controller, or both inside the
  precursor unit's cgroup. Stopping the obsolete unit would therefore kill the
  authoritative successor monitor or the in-flight recovery.
- Cause: `nohup`, `setsid`, and a host-visible detached PID change shell/session
  lifetime but do not move the process out of its inherited systemd cgroup.
- Recovery: inspect `/proc/<pid>/cgroup` for the precursor recovery, every
  successor monitor candidate, and the replacement service before stopping any
  unit. Revalidate and TERM only duplicate host-monitor process groups, prove
  them terminal, then start exactly one TERM-safe successor monitor in a
  dedicated unit and require a fresh healthy check plus one cadence-sleep child.
  If stopping the precursor would terminate the current recovery, run one
  exact-identity cleanup helper in a third cgroup; it must wait for that recovery
  to exit, revalidate the dedicated successor monitor and precursor command,
  then retire only the precursor unit. Never signal the remote trainer.
- Prevention: treat a surviving PID as insufficient monitor isolation evidence.
  Record and compare the successor and precursor cgroup paths, and require a
  dedicated successor unit before removing an obsolete predecessor trigger.
- Observed: the stale PG19 retry6 unit contained the retry6 loop, the current
  Codex recovery, and two retry24 monitor loops (`3087234` and `3155299`) even
  though retry24 itself trained on another TPU. Exact group TERM left the remote
  trainer untouched. Dedicated retry24 unit PID `3605491` then reported healthy
  training at step 415800 and slept on the intended 1,800-second cadence; a
  separate cleanup helper was installed to retire retry6 after the recovery
  controller exits.

### Terminal trainer and mirror children remain as zombies

- Symptom: a completed run reports `checkpoint_mirror_helper_zombie` even
  though W&B is `finished`, the training log reached its configured limit, and
  the terminal checkpoint mirror logged success. The trainer and mirror PIDs
  are `Z*` children of a surviving foreground log-tail wrapper.
- Cause: a monitor can mistake its own remote inspection shell for the trainer
  when it searches the full command text for both the run ID and `train.py`.
  That false `running` classification makes the normally exited mirror zombie
  look like a failed helper.
- Recovery: do not restart or disturb the TPU. Verify the configured terminal
  step, final evaluation, W&B terminal state, checkpoint byte count and
  SHA-256, serialized payload step, and the matching regional completion
  marker. When those agree, record successful completion and never launch a
  retry from the zombie alert.
- Prevention: enumerate processes through `/proc`, require a live non-zombie
  process with an exact `train.py` argv token and the exact run identity, and
  apply live mirror-helper requirements only while that trainer exists. A
  verified terminal checkpoint remains valid after both helper processes exit.
- Observed: FineWeb retry1 completed step `299230`, final validation, and the
  `1674052275`-byte regional mirror with SHA-256
  `7ab355a7df9e5d339337f15973d4444ef3717f548a8d3edc44fecc4094a66e9a`;
  the watcher nevertheless triggered recovery because the foreground wrapper
  had not reaped the two successful children.

### Recovery child inherits the monitor flock

- Symptom: after `MONITOR=CODEX_TRIGGERED`, every later scheduled or manual
  watcher invocation returns `MONITOR=SKIPPED reason=already_running` even
  though no watcher process is active. `fuser` identifies the persistent Codex
  recovery PID as the monitor-lock holder.
- Cause: the watcher launches recovery with `nohup` while flock file descriptor
  `9` remains open. The background recovery child inherits that descriptor and
  therefore owns the lock for its entire session.
- Recovery: retain that exact recovery as the sole orchestrator and do not
  spawn a duplicate merely to regain monitoring. Apply the descriptor fix for
  future children; the inherited lock is released when the current recovery
  exits, after which the installed cron entry can resume.
- Prevention: close the flock descriptor in the recovery child with `9>&-`.
  Test with a deliberately persistent fake recovery process and require a
  second watcher invocation to acquire the lock and reach normal
  `ALREADY_ALERTED` handling rather than `SKIPPED`.
- Observed: FineWeb retry1 recovery PID `1760579` inherited the lock at
  `2026-08-12T13:05:26Z`, suppressing the corrected terminal gate while the
  recovery audit remained active.

### Cron does not define `USER`

- Symptom: a watcher using `set -u` exits with `USER: unbound variable` only under cron, while an interactive check succeeds.
- Cause: cron provides a minimal environment and does not guarantee the interactive shell's `USER` variable.
- Recovery: resolve the remote account from an explicit override, then `${USER:-$(id -un)}`; test once with `env -u USER` before relying on the schedule. This failure occurs before cloud inspection and does not imply that training stopped.
- Recovery: preserve batch/seed/config evidence; validate data and checkpoint, reproduce narrowly, and do not automatically continue a deterministic numerical failure.

### Cron cannot resolve the cloud, tracking, or recovery toolchain

- Symptoms: an interactive monitor works, but cron reports `gcloud: command not found`, the cron-selected Python raises `ModuleNotFoundError: No module named 'wandb'`, or the recovery hook reports `nohup: failed to run command 'codex': No such file or directory`.
- Cause: the monitor relies on the interactive `PATH` and default Python instead of resolving and validating the exact executables required by its cron environment.
- Consequence: the monitor can misclassify a healthy TPU as missing, lose independent W&B evidence, and fail to start recovery. Treat its alert reason as unverified until a direct exact-resource API query succeeds.
- Recovery: pin absolute `gcloud`, W&B-capable Python, and `codex` paths in `ELF_MONITOR_GCLOUD`, `ELF_MONITOR_PYTHON`, and `ELF_MONITOR_CODEX`; fail preflight when an executable or import is unavailable. Execute the exact installed command once under `env -i HOME=<user-home> SHELL=/bin/bash PATH=/usr/local/bin:/usr/bin:/bin` before relying on the schedule. A tool-resolution or monitor-transport failure is monitor infrastructure failure, not evidence of TPU loss, trainer failure, or Spot preemption; require a successful direct exact-resource query before changing the workload classification or starting recovery.
- Observed: `20260804-elf-b-owt-t5-layer-mixing-prenorm-profile-v6e-8-wandb` produced all three errors. Its last successful remote monitor result still showed training at step 2000, while W&B later reached step 5800, proving the initial `tpu_missing_or_unreachable` alerts were monitor failures rather than contemporaneous proof of TPU loss.
- Observed again: the retry5 normalized layer-mixing cron ran at minute 33 with only `/usr/local/bin:/usr/bin:/bin`, reported `monitor_gcloud_unavailable` and W&B unreachable, while the trainer remained healthy. After pinning `/snap/bin/gcloud`, the W&B-capable Python, and `codex`, the cron-like clean-environment check returned `HEALTHY` at step 9000 with W&B running and the 1,674,052,559-byte `checkpoint_7600` mirror verified.

### Recovery passes global Codex options after `exec`

- Symptom: the watcher reports `MONITOR=CODEX_TRIGGERED`, but `codex-recovery.log` immediately contains `unexpected argument '--ask-for-approval' found` and the recorded recovery PID has exited.
- Cause: options such as `--cd`, `--sandbox`, and `--ask-for-approval` are global Codex CLI options in the installed version and must precede the `exec` subcommand.
- Recovery: invoke `codex --cd <repository> --sandbox workspace-write --ask-for-approval never exec <prompt>`, verify the argument ordering with a focused watcher test, then clear only the same run's stale alert marker and rerun the exact cron command so one replacement recovery process is created.
- Observed: the v6e-16 normalized layer-mixing run reached the maintenance threshold while W&B still reported running, but its first recovery subprocess exited on this CLI parsing error.

### Independent monitor installer replaces unrelated cron entries

- Symptoms: a run's historical cron log contains healthy checks, but `crontab -l` no longer contains that run and instead contains another run's watcher; a direct minimal-environment invocation of the missing watcher may still report healthy training.
- Cause: an independent installer replaces the user's complete crontab with a single-run snapshot instead of merging keyed monitor entries.
- Recovery: preserve the current crontab, remove only the exact same-run entry, append the desired entry, install the merged file atomically, verify `crontab -l`, and run the watcher once under cron's minimal environment. Confirm the next scheduled tick as well.
- Prevention: serialize monitor installation and use a shared keyed upsert or managed block. Never replace or delete unrelated cron entries.
- Observed: the retry 3 monitor for `20260805-elf-b-owt-t5-layer-mixing-prenorm-profile-v6e-32-wandb-retry3` was displaced by another run's installer while training remained healthy; its five-minute entry was restored without removing the unrelated watcher.

### Checkpoint cadence aliases with monitor cadence

- Symptom: each scheduled watcher sees a newly finalized checkpoint before its asynchronous mirror marker, so successive normal upload windows accumulate `checkpoint_mirror_pending` failures even though every marker appears seconds later.
- Cause: the checkpoint interval and monitor interval are nearly equal, while the mirror helper polls and uploads after finalization. Sampling the same phase on every cycle turns independent transient observations into a false consecutive-failure threshold.
- Recovery: keep a verified live trainer running. Read the exact checkpoint and marker on the authorized worker, confirm exactly one mirror helper, and run one complete gate after the matching marker appears to clear the transient counter. Do not trigger TPU recovery from a fresh unmirrored checkpoint alone.
- Prevention: include the checkpoint modification epoch in monitor evidence. Treat a missing matching marker as `pending_grace` for a short bounded `ELF_MONITOR_CHECKPOINT_MIRROR_GRACE_SECONDS` window (180 seconds by default) without incrementing the recovery counter, while continuing to report the checkpoint as unverified. After the grace expires, use the normal `checkpoint_mirror_pending` failure path; never accept an older marker or use grace to hide a dead helper.
- Observed: classic retry11 checkpoint 167,201 finalized at `17:51:25Z`; the `17:52:13Z` cron gate observed it before the marker and recorded count 1, then the exact matching marker completed at `17:52:29Z`. Training advanced throughout, and a full gate cleared the counter without recovery.

### Mirror helper children look like duplicate daemons

- Symptom: a healthy run intermittently reports `checkpoint_mirror_helper_count_2`, but an immediate process inspection finds one long-lived mirror daemon and the matching checkpoint marker is valid.
- Cause: `mirror_elf_checkpoints.sh` forks short-lived Bash descendants for command substitutions and pipelines. Those children can inherit the script argv, so a raw `pgrep -f` count mistakes one daemon plus its child for two independent helpers.
- Recovery: keep the verified trainer running and inspect the canonical `OUTPUT_DIR/checkpoint_mirror.pid`. Require exactly one positive PID, a live non-zombie `/proc/<pid>/status`, and an exact cmdline argument sequence containing the mirror script, this output directory, a `gs://` destination, and this run ID. Count only independent matching process-tree roots; ignore descendants and candidates that vanish during inspection. Restart nothing when the canonical root, checkpoint marker, trainer, and next complete gate are healthy.
- Prevention: use the PID file and process-tree root count instead of raw matching-process count. Fail closed for missing, malformed, stale, zombie, identity-mismatched, non-root, or genuinely duplicated canonical helpers, and revalidate liveness and identity immediately before counting each root.
- Observed: the layer-mixing v4 retry1 manual gate at step 172,400 saw two matching argv entries, while the immediate exact-worker check found only canonical helper PID 254830. The 18:30 gate cleared healthy at step 173,600 without recovery; canonical-PID/root validation then passed on both active v4 and v6e runs and on the 18:45 scheduled gate.

### Mirror helper is checked on the process-zero log worker

- Symptom: a healthy multi-worker run reports
  `checkpoint_mirror_helper_missing_pid` after its first checkpoint, while the
  checkpoint object and completion marker already exist in regional storage.
- Cause: automatic JAX process zero can map to a physical worker other than the
  launcher-designated mirror worker. A watcher that uses the process-zero log
  worker for both progress and mirror inspection therefore looks for the PID
  file on the wrong VM.
- Recovery: keep the verified trainer and mirror helper running. Inspect the
  exact process-zero/log worker for progress and the exact launcher-designated
  mirror worker for the canonical PID, helper argv/root identity, local marker,
  and GCS evidence. Do not move or duplicate the helper merely to satisfy a
  single-worker monitor assumption.
- Prevention: configure independent non-negative log-worker and mirror-worker
  indices, use separate SSH results and error logs, and combine only the
  progress/checkpoint candidate from the log worker with helper/marker evidence
  from the mirror worker. Retain one helper and one recovery orchestrator.
- Observed: the PG-19 exhaustive next-eight v5e-64 retry mapped JAX process zero
  and W&B to physical worker 13, while physical worker 0 retained the sole live
  mirror PID and had already mirrored byte- and SHA-256-verified checkpoint
  `100000`. The old watcher escalated twice from worker 13's expected missing
  PID; a split-worker gate returned healthy without restarting training.

## Foreground session and process-lifecycle failures

### Launcher and worker-0 mirror end before the canonical log is visible

- Symptom: the foreground fan-out returns or W&B prints its final sync lines,
  while physical worker 0's mirrored `train.log` has no completed-step metrics;
  another worker may already contain the configured terminal-step sentinel and
  the complete metric series.
- Cause: automatic JAX process zero is not tied to physical TPU worker 0, and
  fan-out/mirror output can be interleaved or lag the process-zero writer. A
  launcher return, W&B finalization, or one physical worker's log is therefore
  insufficient terminal evidence.
- Recovery: do not immediately classify or relaunch. Inspect every worker for
  the exact run identity, live non-zombie trainer, completion/failure sentinel,
  last metric step, checkpoint state, and relevant metric-line count. Identify
  the canonical process-zero log from those facts, copy it locally, and require
  its SHA-256 to match the remote source before reporting the result.
- Prevention: record the physical process-zero/log worker after distributed
  initialization and keep it independent from the launcher-designated mirror
  worker. Terminal monitoring must aggregate all-worker evidence and prefer the
  configured terminal-step sentinel over log-stream timing.
- Observed: the v5e-32 JAX 0.11 gradient-parity retry appeared to end without
  metrics in worker 0's mirror, but physical worker 4 held all 20 comparison
  records and `Reached configured training limit at step 20`. Its local and
  remote log SHA-256 both equaled
  `75d3a68e511be82390abaea9118ca97dd96ddd3779a8a7ef5b36d2c196b3e0b4`.

### Distributed parent abort leaves data-loader children holding VFIO

- Symptom: the intended trainer parent exits during distributed compilation,
  but multiple current-run Python children remain on every host with PPID 1 and
  inherited VFIO or libtpu-lock descriptors. A naive process-name check can
  mistake them for an independent healthy workload or an idle device.
- Cause: forked data-loader children outlive an abruptly terminated distributed
  parent and retain accelerator-related file descriptors inherited from that
  process tree.
- Recovery: preserve the failed attempt and prove each survivor's numeric owner,
  PID, PPID, start time, full command, and exact immutable run identity. Only
  under the repository's current-user stuck-run authorization, send TERM to
  those exact children, revalidate identity before any KILL escalation, and
  require every child, VFIO holder, and canonical lock holder to clear on every
  worker. Quarantine only proven current-run, zero-byte, unheld lockfiles under
  unique names; never touch foreign, unknown, healthy, checkpoint, output, or
  source state.
- Prevention: after any first-step/coordinator abort, audit the full process tree
  and file-descriptor holders rather than checking only the original parent PID.
- Observed: the failed v5e-32 JAX 0.6.2 parity attempt left eight children per
  host across eight hosts. All 64 exact-identity children exited on TERM, no
  KILL was required, and all five VFIO resources per worker were then idle.

### Foreground fan-out re-creates an intentionally stopped trainer

- Symptom: after the exact remote trainer and wrapper are stopped, the still-open local `gcloud compute tpus tpu-vm ssh` fan-out executes the remote launch command again; a reused output directory may then auto-resume from its newest checkpoint.
- Cause: the controlling foreground fan-out remains alive while only its remote command processes are terminated.
- Recovery: close the exact local foreground launch session first, then revalidate and stop only the superseded remote trainer, wrapper, and checkpoint mirror. Verify the old process stays absent and accelerator occupancy is `AVAILABLE` before launching a distinct run identity.
- Prevention: for an intentional stop, remove its recurring monitor, close its foreground launcher, and only then terminate and verify the remote processes. Never reuse the stopped run's output path for a scientifically changed retry.

### Log-tail wrapper leaves a successful trainer zombie and triggers a duplicate SSH retry

- Symptom: training reaches its configured terminal step and W&B finishes, but
  the foreground all-worker SSH fan-out remains open. The mirror worker shows a
  zombie trainer parented by `tail --pid`; terminating that tail closes SSH,
  after which `gcloud` retries the same remote launch command and attempts to
  start the completed run identity again.
- Cause: the launch shell used `exec tail --pid=<trainer-pid>`. Replacing the
  shell with `tail` left no parent to `wait` for and reap the trainer, so the
  trainer PID remained as a zombie and `tail --pid` never observed its
  disappearance. The remote launch command also lacked an atomic launch-once
  marker, so a transport retry could execute the trainer start twice.
- Recovery: first prove completion from every worker, the canonical
  process-zero log, W&B, and checkpoint state. Stop the exact local foreground
  launch session before touching remote processes. For the completed exact
  `kristine` run, revalidate the mirror tail's owner, PID, parent, full command,
  and run identity, TERM only that stale tail, and confirm trainers, mirrors,
  and VFIO holders are absent on every worker. Preserve the duplicate-attempt
  log and output; never relaunch or resume that completed identity.
- Prevention: atomically acquire a per-worker in-progress claim before fallible
  preparation, remove it only on a proven pre-spawn exit, and persist a distinct
  committed marker only after a detached node-local supervisor and required
  checkpoint mirror have spawned and their PIDs are atomically published. The
  supervisor owns and reaps the trainer and atomically records its status, so
  status persistence does not depend on the SSH-bound shell surviving. An
  unresolved claim is indeterminate and must fail closed. On the log worker,
  run `tail -F` separately under exit and signal cleanup traps; defer a signal
  exit until the newly forked tail PID has been captured, then stop and reap
  that exact PID. Retain the pre-commit signal traps across committed-marker
  publication until the post-commit handlers replace them, so no default-signal
  window is externally visible. After live-tail cleanup, replace the cleanup
  handlers directly with deterministic terminal handlers until status reporting
  exits; do not restore default signals while terminal work remains. After
  supervisor readiness, accept either fully validated
  live processes or a valid atomically published terminal status, polling
  across the fast-exit publication boundary. A committed
  replay returns the stored terminal status, succeeds only when the live
  supervisor and trainer match their complete expected argument vectors, and
  otherwise fails indeterminate. The checkpoint mirror must treat the same
  atomic status as terminal evidence and run its final scan even when it never
  sampled the trainer live. Until that status exists, validate the atomically
  published supervisor PID, process-generation start time, and readiness
  evidence; do not infer liveness by
  searching for a run ID that a fresh trainer argv may not contain.
  Regression-test marker transitions through
  injected failure and replay, launcher loss after commit, zero and nonzero
  terminal statuses, argument mismatch, failed status persistence with tail
  cleanup, and the absence of `exec tail --pid`.
- Observed: the completed v5e-16 b128/BKV512 MFU gate reached step 800 and W&B
  `finished`, but its mirror worker retained the trainer as a zombie under
  `tail --pid`. After the exact tail was terminated, the foreground `gcloud`
  fan-out retried on worker 0. W&B `resume: never` rejected the duplicate run
  ID before TPU initialization. The local retry loop was stopped, all four
  workers were audited clean, and the successful run artifacts were preserved.

### Foreground SSH session closes

- Symptoms: local SSH exits; distributed runtime later enters shutdown, `ExitTimeoutWatcher` sends `SIGABRT`, and Apport writes large core dumps.
- Cause: losing the required foreground fan-out can initiate distributed shutdown even if remote Python briefly survives.
- Recovery: preserve the launch session for the run lifetime, ignore `SIGHUP` in the remote wrapper, disable core dumps, and write logs node-locally. Do not mistake temporary `D`/`do_exit` during core dumping for a TPU kernel lock.

### Process exits but TPU remains healthy

- Causes: deterministic code/config/dependency failure, OOM, signal, or external workload collision.
- Recovery: capture exit/log evidence, verify newest checkpoint, diagnose before retry, and reuse the TPU only after all holders clear.

### Process exists but no progress

- Causes: compilation, checkpoint write, evaluation, distributed deadlock, filesystem pressure, or stuck shutdown.
- Verify: newest metric/log time, process state/wait channel, device holder, disk, checkpoint temporary files, and TPU state/health.

## Checkpoint and artifact failures

### Globally sharded pmap state is indexed during multihost checkpoint save

- Symptom: training reaches a scheduled checkpoint step, then process zero
  raises a JAX runtime `RET_CHECK` requiring the device assignment to include
  all slice devices. The traceback ends at checkpoint unreplication such as
  `tree_map(lambda x: x[0], state)`; peer hosts subsequently report only
  shutdown-barrier failures.
- Cause: JAX 0.11 pmap outputs use a globally sharded leading replica axis. In
  a multihost slice, integer indexing with `x[0]` requests a distributed
  gather/reshard even though every replica is scientifically identical. A
  limited-ICI topology can reject that synthesized device assignment.
- Recovery: preserve the failed attempt and audit every worker before deciding
  whether a valid checkpoint survived. A non-process-zero worker and its mirror
  can finish the same checkpoint even when process zero fails its extraction;
  accept it only after completion-marker, byte-count, SHA-256, and serialized
  payload-step verification. For a fresh retry, extract each replicated leaf
  from `array.addressable_shards[0].data.squeeze(0)` before `jax.device_get` so
  no cross-host reshard is compiled. Keep all scientific settings unchanged.
- Prevention: make global-array integer indexing fatal in a focused checkpoint
  test, then gate the first scheduled save on the exact multihost JAX runtime.
  After a parent abort, audit and clean exact current-run data-loader children
  through the guarded orphan protocol before reusing the TPU.
- Observed: the FineWeb full-window v5e-32 FP32-all-reduce continuation trained
  through finite step `187500`, then JAX 0.11 failed at `x[0]` on physical
  process-zero worker 1. Seven peers produced identical checkpoint bytes and
  worker 0 durably mirrored a `1,255,708,131`-byte, payload-step-187500 object
  with SHA-256
  `b317061f9edeabb01bee29e57bacc969ed62e3847c1cd103fd4586e1276f7765`.

### v5e fused-Pallas scoped VMEM exhausts before the first step

- Symptoms: `pallas_flash_fused` raises `XlaRuntimeError: RESOURCE_EXHAUSTED`
  during `model.init` or the first train-step compile even though host RAM, HBM,
  disk, runtime, dataset, TPU health, and VFIO gates pass.
- Confirmed signatures for ELF-B at length 1024, three 256-dimensional heads,
  and four examples/chip: `BQ1024/BKV256` forward/model-init requested
  `17.14 MiB` versus a `16.00 MiB` scoped-VMEM limit; `BQ512/BKV256` model-init
  passed but backward requested `23.39 MiB` for full per-device-batch Q/K/V
  gradients.
- Recovery: preserve the zero-step attempt and use a fresh identity. Move QKV
  projection and gradients outside the custom-call boundary with non-fused
  `pallas_flash`, then gate tiles without changing model science or effective
  batch. For the observed workload, `BQ1024/BKV512` completed and was fastest.
- Prevention: first gate both model initialization and a complete training step
  on the exact v5e runtime, geometry, and per-chip batch. A forward-only smoke
  or a successful v6e fused kernel is not proof that v5e backward will compile.

### Obsolete precursor recovery would duplicate an active successor

- Symptoms: a predecessor monitor or recovery prompt proposes checkpoint localization, a capacity query, or a fresh retry even though a later retry in the same scientific lineage is already active; the proposed checkpoint is older than the successor's current step.
- Cause: recovery treated the alerting report as canonical without reconciling all experiment reports, active installed monitors/recovery orchestrators, and authenticated read-only W&B lineage.
- Recovery: stop before checkpoint copying, capacity acquisition, staging, or launch. Match scientific configuration, retry/resume ancestry, checkpoint step, W&B run ID, timestamps, state, and latest metric step to select one canonical lineage head. If a newer active successor exists, retire only the obsolete predecessor monitor and recovery trigger; preserve the successor's monitor, trainer, artifacts, and capacity, and record the reconciliation in both reports.
- Prevention: make lineage reconciliation the first preemption-recovery gate, before capacity or checkpoint work. Never create a duplicate full run from an older checkpoint, and never infer that no successor exists from one stale report or one W&B `crashed` label.

### Checkpoint write fails or remains `.tmp`

- Causes: insufficient disk, interrupted atomic rename, serialization error, or preemption.
- Recovery: never resume from `.tmp`; verify finalized checkpoint name, size, and integrity before recording it valid.

### Mirror uploads incomplete checkpoint

- Symptom: `.tmp` checkpoint appears in GCS.
- Cause: broad glob mirrors a file before atomic finalization.
- Prevention: mirror only names matching `^checkpoint_[0-9]+$`; require a success marker after upload.

### Mirror fails

- Causes: mismatched bucket region, missing permissions/service account, absent `gcloud`, network failure, insufficient object storage access, or an interpreter incompatible with that exact Cloud SDK installation. The ELF training interpreter and Cloud SDK interpreter are separate compatibility domains; neither the ELF runtime, image system Python, nor SDK bundled default is universally correct.
- Signatures: legacy Cloud SDK 127 under Python 3.12 fails while importing the removed `imp` module. Some refreshed Cloud SDK installations instead fail through system Python 3.8 dependencies with `TypeError: 'type' object is not subscriptable`, while the node-local ELF Python 3.11 successfully runs the same storage command.
- Recovery: keep training if node-local checkpoints are valid, record mirror failure, and never fall back to a cross-region/default bucket. Verify no mirror helper remains. On the exact worker/image, preflight the exact `gcloud storage` command shape, authentication, and intended regional bucket or prefix under candidate interpreters; run the same subcommand and flags against a harmless run-scoped probe instead of accepting `--help` alone. Never blindly set `CLOUDSDK_PYTHON` to `ELF_RUNTIME_PYTHON`; choose an explicitly verified interpreter, or leave `CLOUDSDK_PYTHON` unset only after the bundled SDK default passes the same probe; start exactly one replacement helper with that choice and require the checkpoint completion marker plus remote-size verification.
- Prevention: record the Cloud SDK version and resolved Python path/version; fail closed unless the exact storage probe succeeds on every checkpoint-writing worker. Monitor helper liveness and marker completion after every finalized checkpoint, and repeat the probe after an image, SDK, or runtime change.

### Resume checkpoint missing or inconsistent

- Symptoms: missing file, different sizes/hashes across workers, deserialization error, wrong optimizer/model tree, or step mismatch.
- Causes: node deletion, partial mirror, incompatible code/config, or incorrect path.
- Recovery: refuse resume until verified. If no valid checkpoint survives, record that fact and restart the exact configuration from step zero.

## Evaluation and generation failures

### Validation consumes a training-only encoder mask

- Symptom: an exact checkpoint restores and the fixed prompt digest passes, but the first validation batch raises `KeyError: encoder_attention_mask`; no sampling arm completes.
- Cause: the validation step expects a derived field that the training step constructs after the dataloader boundary, while evaluation receives only the concrete dataloader fields `cond_seq_mask` and `attention_mask`.
- Recovery: derive the encoder mask on device from the concrete validation batch exactly as training does, add focused validation-step coverage using the real dataloader shape, and use a fresh evaluation identity. Do not alter the scientific sweep configuration.
- Observed: FineWeb retry-5 checkpoint 3.0M and 3.5M retry 3 restored correctly but completed zero arms before this boundary fix; retry 4 passed validation and began generation.

### Failed evaluation is finalized as successful in W&B

- Symptom: the local evaluation exits with an exception and zero completed arms, but W&B records state `finished`.
- Cause: unconditional normal finalization runs without an exceptional exit code.
- Recovery: finalize the W&B run with a nonzero exit code on the exception path, preserve the failed attempt as infrastructure-invalid, and retry under a fresh run identity. Verify the failure path with a focused test.

### Checkpoint identity logging touches JAX before distributed initialization

- Symptom: evaluation fails before checkpoint restoration when a pre-initialization identity log calls a helper that queries JAX process state.
- Cause: the logging helper is safe only after distributed JAX initialization, but checkpoint preflight runs earlier.
- Recovery: log the already known physical worker identity directly with standard Python logging before initialization; reserve JAX-backed process-zero helpers for the initialized phase. Test the checkpoint-evaluation entrypoint with initialization ordering preserved.

### Fresh-output guard counts launcher bookkeeping as evaluation output

- Symptom: a fresh evaluation output directory is rejected before restore because it contains only launcher-created `launch_training.sh` and `train.pid`.
- Cause: the output guard treats launch-control files as scientific artifacts.
- Recovery: exclude the exact known launcher bookkeeping files while continuing to reject checkpoints, generated samples, metric artifacts, and all unknown entries. Cover both allowed bookkeeping and rejected scientific artifacts with focused tests.

### All decoded generations are empty or whitespace-only

- Symptom: the requested sample artifact has the exact expected row count, but `empty_sample_rate` is `1.0`; GPT-style perplexity is skipped and MAUVE reports that it requires at least two nonempty generations.
- Distinguish model collapse from a short/empty artifact or EOS bug: require the saved-row count, raw pre-mask model-token count, EOS rate, per-sample model-token length, decoded examples, TPU/process health, and post-evaluation training progress. Full-length sequences with `eos_rate: 0.0` that decode entirely to whitespace are a one-token generation collapse, not missing generation work and not a MAUVE-backend failure.
- Interpret objectives separately: held-out decoder CE on noised ground-truth latents and held-out denoising L2 can both improve while terminal sampler latents collapse into one decoder decision region. Record this as evidence of end-to-end rollout or train/inference latent-distribution failure; do not claim the exact mechanism without checkpoint-specific latent/token diagnostics.
- Recovery: preserve the exact sample and metric artifacts, report PPL and MAUVE as undefined rather than zero or a prior-step value, keep a healthy trainer running, and inspect the next scheduled evaluation. A diagnostic retry should record raw token-frequency concentration and compare decoding of ground-truth/noised latents with sampler terminal latents; do not add concurrent TPU capacity without authorization.
- Prevention: gate every scheduled evaluation on exact sample count, nonempty rate, dominant-token concentration, EOS/max-length behavior, repetition, and PPL/MAUVE availability. Treat low CE or low PPL on a degenerate nonempty sample set as insufficient evidence of language quality.
- Observed: the layer-mixing retry1 step-50k samples were already dominated by repeated `s` (`repetition_4gram_rate=0.9801`, MAUVE `0.00407`). At step 100k, all 1,000 samples decoded to whitespace despite improving held-out L2/CE; all raw sequences had 1,012 tokens and no EOS, so PPL and MAUVE were correctly undefined while training resumed.

## Monitoring and Spot failures

### Immutable watcher snapshot has an older command-line contract

- Symptom: a newly installed monitor exits with usage text before any cloud inspection, while the same watcher invocation works from the current repository.
- Cause: the immutable evaluation snapshot contains an older `watch_elf_run.sh` positional-argument contract that lacks the evaluation workload-role argument used by the new wrapper.
- Recovery: keep the immutable training/evaluation source untouched and point the local recurring wrapper at the tested current-repository watcher with an absolute path. Run the exact wrapper once under cron's minimal environment before installation. This is monitor infrastructure failure, not workload failure.

### Recovery follows a stale precursor despite a newer live successor

- Symptom: a recovery controller begins capacity selection or checkpoint transport for an interrupted precursor even though a newer successor in the same experiment lineage is already training, has newer evaluations/checkpoints, and has an installed recurring monitor.
- Cause: recovery state was reconciled only against the stale precursor summary instead of the full successor-lineage reports, W&B identities/states, durable checkpoint evidence, and installed monitor entries.
- Recovery: stop before creating any duplicate trainer, W&B run, or output identity. Preserve every copied checkpoint, remove only the obsolete precursor monitor, and continue monitoring the authoritative successor. A valid old checkpoint is not a valid recovery source when a newer live successor exists.
- Prevention: before capacity requests, cross-region checkpoint copies, localization, or relaunch, require one lineage reconciliation that inventories all related reports, W&B run IDs/states, newest verified checkpoints/evaluations, and installed monitor keys. Resolve the training runtime separately from the Cloud SDK interpreter, and preflight the exact `gcloud` command under its selected interpreter before all-worker fan-out.
- Observed: retry5 checkpoint 15,200 was copied from `us-east5` to `us-central2` and localized only on worker 3 before reconciliation found the live v4-8 retry1 successor at step 150,000+ with W&B `6mapl0ve`. Workers 0-2 had failed before writing because legacy `gcloud` 127 was forced under Python 3.12 and imported the removed `imp` module. Recovery stopped before any duplicate launch; no files were deleted, and only the successor plus classic retry11 monitors remained installed.

### Spot preemption or TPU deletion

- Symptoms: state changes to `PREEMPTED`/`DELETING`, SSH refuses connection, or direct API returns `NOT_FOUND`.
- Recovery: record last step/checkpoint, stop treating node-local artifacts as available after deletion, use equivalent authorized `kristine` Spot replacement, stage identical source/data/config, and resume from the newest verified region-local mirror if one exists.
- Observed: `20260804-elf-b-owt-t5-layer-mixing-prenorm-profile-v6e-8-wandb` ended after W&B step 5800; its exact Spot TPU later returned `NOT_FOUND`, and its durable checkpoint prefix contained no objects. The available audit query did not identify the deletion actor, so record resource disappearance as confirmed and Spot reclaim only as the leading hypothesis.

### Health becomes `TIMEOUT` or unhealthy

- Recovery: reject the node without reboot. Continue equivalent replacement monitoring.

### Monitor sees unchanged state

- Interpretation: unchanged capacity or progress is a monitoring condition, not completion and not authorization to launch a duplicate query/run.

### Logging/tracking outage

- Symptoms: W&B/network errors while local training continues.
- Recovery: distinguish telemetry failure from training failure. Preserve node-local logs/checkpoints and report tracking status separately.

### Pre-JAX W&B owner differs from distributed process zero

- Symptoms: W&B initializes on physical worker 0, then reports `crashed` with
  zero history and a stale heartbeat while a different physical worker holds
  the canonical process-zero log and all distributed trainers keep advancing.
- Cause: W&B ownership was chosen before `jax.distributed.initialize()` resolved
  the process index. Post-initialization metric calls are process-zero gated, so
  the pre-JAX owner can receive no history or heartbeat-producing activity.
- Recovery: keep the exact healthy trainer running. Inspect all workers, identify
  the process-zero log worker from advancing metrics, and monitor it separately
  from the launcher-designated checkpoint-mirror worker. Report W&B as failed
  telemetry and ignore its terminal state only while the exact trainer is live
  and advancing; never relaunch from the W&B label alone.
- Prevention: initialize W&B only after distributed JAX setup and only on the
  resolved process-zero worker, or keep the launcher-designated physical worker
  zero as the canonical telemetry owner after initialization. In that physical
  ownership mode, route application logs, progress output, validation artifacts,
  and W&B metrics through `ELF_TPU_WORKER_INDEX` while leaving scientific JAX
  sharding on `jax.process_index()`. Add a multihost test whose physical worker
  zero is not JAX process zero and require W&B history plus heartbeat
  advancement.
- Observed: the FineWeb full-window v5e-32 FP32 recovery assigned W&B to physical
  worker 0 and JAX process zero to physical worker 1. W&B `jhu2i0bs` stopped its
  heartbeat after 49 seconds with zero history and became `crashed`, while the
  canonical log advanced beyond step 130460 and the TPU remained READY/HEALTHY.

## Postmortem template

Append new failure classes to the appropriate section above and use this structure in the attempt report:

```markdown
### <failure name>

- Attempt / retry:
- TPU / zone / topology:
- Stage reached and last step:
- First error and exit status:
- Detailed log path:
- Confirmed root cause and evidence:
- Alternative causes rejected:
- Packages before / after:
- Repair command:
- Verification command and result:
- Checkpoint / artifact state:
- Recovery action:
- Prevention gate added:
```

Only call a root cause confirmed when direct evidence supports it. Label incomplete explanations as hypotheses and keep monitoring/recovery state separate from the postmortem.

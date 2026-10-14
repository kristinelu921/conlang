# AGENTS.md

This file defines the repository-specific expectations for coding agents. Follow
these instructions for all work in this repository unless the user explicitly
requests otherwise.

## Code Style

- Add type annotations to function arguments and return values.
- Use descriptive, fully spelled-out names for variables and functions. Avoid
  abbreviations unless they are standard terms in the domain.
- Give non-trivial functions a concise, one-line docstring that states their
  purpose.
- Prefer cohesive functions over many small wrappers. Do not extract a helper
  solely to hold two or three straightforward lines. A small helper is
  appropriate when it removes meaningful duplication, isolates complex logic,
  or provides a useful named abstraction.
- Trace values back to their producer and inspect their concrete runtime shape
  before writing type-dependent code. Determine the single concrete type the
  producer is expected to return, then prefer one `assert isinstance(...)` at
  the boundary over `if`/`elif`, `match`, or fallback branches that cover every
  theoretically possible type. After that assertion, implement the expected
  shape directly and operate under the assumption that it is the input type.
- Use that single boundary assertion for producer-shape invariants; do not add
  repeated nested type assertions. Do not add redundant assertions around
  ordinary typed access or trusted library operations.
- Prefer modifying an existing implementation or adding a well-scoped flag or
  configuration option when it can cleanly support the requested behavior.
  Create a new script or file only when it represents a distinct
  responsibility, provides meaningful reuse, or is clearer than extending the
  existing code.
- Add comments only when they explain non-obvious behavior, an important or
  unexpected discrepancy, or the reason behind an implementation choice.
- When code implements or differs from a research paper, add a one-line comment
  containing the paper title and a link to it.

Use this general shape for new functions:

```python
def descriptive_function_name(typed_argument: ArgumentType) -> ReturnType:
    """State the function's purpose in one concise line."""
    descriptive_variable_name = ...
    return descriptive_variable_name
```

When a producer's static type is broader than its concrete supported output,
use this shape instead of exhaustively branching over all declared types:

```python
def count_tokens(producer_output: object) -> int:
    """Count tokens in the producer-supported list shape."""
    assert isinstance(producer_output, list)
    return len(producer_output)
```

## Testing

- Add focused tests for every behavior change and bug fix.
- During a refactor, establish the current behavior with tests before changing
  the implementation. Run the same tests after the refactor and confirm that
  outputs match unless the requested change intentionally alters them.
- Test at the narrowest useful level, then add integration or end-to-end
  coverage when the change crosses module boundaries or affects a complete
  workflow.
- Run the relevant tests before reporting that work is complete. Report any
  tests that could not be run and the reason.

## Presenting Data

When the user asks to see data, create a clear, labeled visualization and serve
it locally. Provide the localhost URL and keep the server running so the user
can open it. Choose a plot appropriate to the data and include units, legends,
and titles where applicable.

## Experiment Logging

For every experiment:

- Create or update a log under `experiment_log/`, organized by date.
- Record the complete experiment configuration, including random seeds and the
  exact command used to run it.
- Clearly identify hyperparameters that changed relative to the baseline or
  previous run.
- Record the resulting metrics, relevant artifacts or checkpoint paths, and
  whether the run completed successfully.
- Do not overwrite prior experiment records; append a new entry or create a new
  uniquely named file.

## Run Continuity

- Before staging or launching any new or retry ELF run, commit the smallest
  reproducible configuration closure for the exact selected run, push that
  commit to the run's branch, and open or update a pull request containing it.
  Normally this is one primary run YAML: the exact file passed to the launcher
  and later linked to the launched, fully monitored run. Include another YAML
  only when that primary file directly depends on it and the required version
  is not already committed. Do not bulk-stage configuration directories or all
  modified/untracked `*.yml` or `*.yaml` files, and do not add unrelated,
  abandoned, superseded, or speculative candidate configs merely because they
  exist in the worktree. Add a retry config only when it is selected for
  immediate launch under its own run identity, and retain it in the finalized
  pull request only when that run is launched and monitored. Before every config
  commit, inspect `git diff --cached --name-only` and reject any YAML outside
  that minimal closure. Record the commit SHA, pull-request URL, primary config
  path, and launched run identity in the experiment log; never launch from an
  uncommitted or untracked run configuration.
- On every user request to launch, resume, relocate, or update an ELF run, first run `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`. Select only from its final `AVAILABLE` table and only a `kristine` resource, unless the current conversation contains the TPU Ownership Safeguard key phrases for a read-only sweep and a later idle acquisition. Choose the most efficient compatible hardware using measured/validated throughput, HBM headroom, topology compatibility, and region-local checkpoint/data availability; do not select merely by chip count.
- A single finalized checkpoint may be copied once between regional buckets when required to use the selected optimal TPU. Verify its source completion marker, object size, and SHA-256 before the copy and again after node-local localization. Never copy a dataset across regions. First look for a complete, verified dataset at the selected TPU region's `kristine` bucket path and stage/reuse it; redownload only when that regional `kristine` path is absent or fails integrity verification.
- Before rejecting a selected TPU for occupancy or disk, identify each holder's owner, command, and run identity. Leave a healthy active selected ELF run alone. Standing user authorization permits terminating a stuck or race-blocked current-user ELF process only on an exact `kristine` TPU, after proving its owner, PID, full command, and exact run identity; record the stale-progress evidence, send TERM, revalidate identity before any KILL escalation, then confirm every VFIO resource is idle. Never apply this exception to a foreign, unknown, non-ELF, or healthy process. If disk is insufficient, first prove the selected run's verified dataset is absent. Only then may the agent remove files that are all: current-user owned, inactive, explicitly identified as irrelevant to every active run, regenerable staging/download/cache artifacts, and recorded before removal. Never remove checkpoints, outputs, source snapshots, credentials, or any other user's files.
- Before every launch, snapshot the current ELF worktree with
  `tpu_scripts/run_staging.sh` under
  `/kmh-nfs-ssd-us-mount/staging/kristine/elf/<run-id>`. Copy that exact
  snapshot to every TPU worker's node-local storage and train only from the
  VM-local copy; never launch directly from the mutable repository or NFS
  staging path.
- Whenever the user asks to launch a run, keep a recurring monitor active for
  that run and check it at least every 30 minutes. Do not treat a successful
  launch as the end of the task.
- Continue monitoring through transient failures, TPU preemptions, capacity
  shortages, and recovery attempts until the run completes successfully or the
  user explicitly asks to stop it.
- At every check, verify the TPU's current state and health, the training
  process, recent progress or metrics, and the newest valid checkpoint. Append
  material status changes and recovery actions to the experiment log.
- Do not infer success or failure from the foreground launcher's return, W&B
  finalization, or physical worker 0's mirrored log alone. On multi-host JAX,
  process zero can map to another physical worker. Before assigning terminal
  status, inspect every worker for the exact run identity, completion/failure
  sentinel, last metric step, and checkpoint state; identify the canonical
  process-zero log by evidence, then preserve and checksum it.
- If the training process exits while the TPU remains usable, diagnose the
  failure and safely resume the same run on that TPU from its newest valid
  checkpoint when possible.
- If the TPU is preempted, disappears, becomes unusable, or cannot host the
  continuation, rerun `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh
  he-vision-group` and select replacement capacity only from its final
  `AVAILABLE` table. Populate every replacement worker with the same source,
  effective configuration, and region-local data, and resume from the newest
  valid checkpoint. On every such recovery, re-push the immutable source
  snapshot and effective configuration to every replacement worker before
  checkpoint localization or launch; never rely on files from a prior TPU.
  Keep monitoring after every relaunch and keep repeating the filtered audit
  when no existing resource is suitable.
- Treat the user's launch request as standing authorization only for selecting
  and using an existing `AVAILABLE` `kristine` TPU, plus staging and relaunching
  that exact run. It does not authorize creating new TPU resources. When an
  existing matching v6e is unavailable, an existing `AVAILABLE` v5p-8, v5p-16,
  or v5p-32 is also an approved replacement for the same run, subject to the
  region-local checkpoint/data rule and fresh compatibility, HBM, and throughput
  gates. A v5p fallback requires a fresh retry identity and may use only a
  documented parameter-free backend substitution required for compatibility. It
  does not authorize larger or additional experiment capacity, unrelated
  configuration changes, deletion, rebooting, or use of a TPU whose name lacks
  `kristine`. A non-`kristine` TPU requires the separate key phrases in TPU
  Ownership Safeguard, and an active run on any TPU remains off limits.
- Keep one recovery/query orchestrator active and do not create duplicate runs
  or exceed the run's originally approved active capacity. Capacity errors and
  unchanged state are monitoring conditions, not completion conditions.

## TPU Ownership Safeguard

- Always prefer a TPU or TPU VM whose exact resource name contains `kristine`,
  case-insensitively. Choose a suitable `kristine` TPU before considering any
  other resource. Idle, ready, same-project, collaborator-created, or
  technically accessible status does not by itself authorize another owner's
  TPU.
- A wide sweep of all other TPUs is allowed only when the user's current
  message contains the exact key phrase `AUTHORIZE WIDE READ-ONLY TPU SWEEP`.
  That phrase is the specification that permits the sweep. A launch, resume,
  relocate, recovery, or capacity request does not contain this permission and
  does not imply it.
- The authorized sweep is read-only. Its only purpose is to determine which
  TPUs are not actively held. Read cloud state and a minimal occupancy check.
  Do not write, signal, lock, reboot, reset, reimage, copy files, start a
  workload, or leave a session connected.
- Reading another person's TPU is possible under that phrase. It is never
  acceptable to interrupt a run, affect a run, or make yourself known to a run.
  Once a read shows the TPU is actively held, end that visit: do not announce
  it, and do not touch the job's processes, logs, checkpoints, or devices, or
  take any action the running workload can observe. A later full sweep may
  read-only re-query a TPU previously found actively held, because its job may
  have finished and the TPU may be inactive again. Every sweep stays read-only
  and must never touch an existing active job.
- On a non-`kristine` TPU, do not edit, stop, or touch an actively held run
  owned by somebody else at any point, even when the user requests it. No later
  message, key phrase, launch, recovery, or debug request waives that ban. It
  covers processes, logs, checkpoints, configs, devices, and any other file or
  state the run is using. A process owned by kristine may be terminated only
  when the user explicitly authorizes that termination, and never as part of a
  sweep.
- If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop,
  touch, or interrupt it. On a `kristine` TPU, the documented recovery rules
  apply.
- When the read shows a TPU is not actively held, leave it untouched until the
  user gives a separate explicit permission. Acquisition requires the exact key
  phrase `AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION` together with that TPU's
  exact resource name, in the current conversation, and only for a TPU the
  read-only check showed is not actively held. Every actively held TPU stays
  untouched.
- Without `AUTHORIZE WIDE READ-ONLY TPU SWEEP`, do not ask for a TPU name and
  do not inspect another owner's TPU. Keep using
  `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`
  on the monitoring cadence until an existing compatible `kristine` resource
  appears. Never invoke `tpu_scripts/query_tpus.sh` unless the user explicitly
  requests new TPU provisioning in the current conversation; a generic launch,
  resume, relocation, recovery, or persistence request is not provisioning
  authority.
- Continue to require explicit user authorization for state-changing operations
  on `kristine` TPUs. The narrowly scoped standing recovery authorization under
  Run Continuity counts as explicit authorization only after the user asks to
  launch that run. It does not authorize provisioning or other mutations merely
  because a name contains `kristine`. The two key phrases above are separate
  from that standing authorization.
- Before launching on a READY `kristine` TPU, inspect `/dev/vfio/*`. A foreign,
  non-`kristine` VFIO holder on that `kristine` TPU may be terminated and
  claimed without rebooting the TPU. A current-user `kristine` ELF holder may
  be terminated only under the documented stuck/race recovery exception above.
  Process termination of someone else's run stays on `kristine` TPUs. An
  authorized read of a non-`kristine` TPU never includes terminating, signaling,
  or otherwise disturbing its processes.


## Session Notifications

- Send an ntfy notification to `https://ntfy.sh/kluping` whenever the session
  needs the user to answer a question or provide input. Start every such
  notification with `APPROVAL REQUIRED:`.
- Send an ntfy notification to the same topic whenever a meaningful task or
  milestone is completed.
- Keep the notification concise and include the repository name plus a short
  description of the question or completed milestone. For example:
  `curl -fsS -d "elf: <question or milestone>" https://ntfy.sh/kluping`.
- Treat notification delivery as best effort: do not expose secrets in the
  message, and do not block the task if ntfy is temporarily unavailable.

## Completion Checklist

Before finishing a task:

1. Confirm the implementation follows the code-style rules above.
2. Run the relevant tests and compare behavior where required.
3. Log any experiments that were run.
4. Summarize changed files, test results, and any remaining limitations.

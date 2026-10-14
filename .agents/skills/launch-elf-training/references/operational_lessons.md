# ELF launch, evaluation, and recovery operational lessons

Read this file for every ELF TPU launch, exact-checkpoint evaluation, or
recovery. It consolidates confirmed failure memory and latency reductions. The
parent `SKILL.md`, repository `AGENTS.md`, and ownership guard remain
authoritative when a rule overlaps.

## Contents

- [Fast path that remains fail-closed](#fast-path-that-remains-fail-closed)
- [Select checkpoints from retained reality](#select-checkpoints-from-retained-reality)
- [Preserve paired-ablation validity](#preserve-paired-ablation-validity)
- [Hardware and compilation lessons](#hardware-and-compilation-lessons)
- [Integrity and dependency failure memory](#integrity-and-dependency-failure-memory)
- [Exact-evaluation monitoring](#exact-evaluation-monitoring)
- [Reproducible reporting and artifact handoff](#reproducible-reporting-and-artifact-handoff)

## Fast path that remains fail-closed

1. Resolve the scientific unit first: fresh training, exact resume, or
   evaluation-only. Record the baseline, treatment, seed, requested artifacts,
   and what must remain paired before choosing hardware.
2. Run the server-side `kristine`-filtered `check_tpus.sh` workflow and use only
   its final `AVAILABLE` table. Set `ELF_TPU_INITIALIZE_MISSING=0` during a
   read-only audit. If no compatible candidate exists, repeat that filtered
   audit on the monitoring cadence. A wide sweep of other TPUs requires the
   exact current-message phrase `AUTHORIZE WIDE READ-ONLY TPU SWEEP` and stays
   read-only: never interrupt a run, affect a run, or make yourself known to a
   run. A later full sweep may read-only re-query a TPU previously found
   actively held, because its job may have finished and the TPU may be inactive
   again. Every sweep stays read-only and must never touch an existing active
   job. On a non-`kristine` TPU, do not edit, stop, or touch an actively held
   run owned by somebody else at any point, even when the user requests it. A
   process owned by kristine may be terminated only when the user explicitly
   authorizes that termination, and never as part of a sweep. If a run is owned
   by somebody else on a non-`kristine` TPU, never edit, stop, touch, or
   interrupt it. On a `kristine` TPU, the documented recovery rules apply. Acquiring an idle non-`kristine` TPU requires the later phrase
   `AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION` with that exact resource name.
   Never invoke `tpu_scripts/query_tpus.sh` unless the user explicitly requests
   new TPU provisioning in the current conversation. After selecting one
   candidate, use exact-name describe and occupancy calls instead of repeatedly
   sweeping the inventory.
3. Prefer a single-host topology for generation while the current generation
   path gives every host the full requested sample count. A larger multi-host
   topology can duplicate samples and add distributed launch-ID risk instead of
   increasing useful generation throughput. Use multi-host only after proving
   work is sharded and artifacts are aggregated without collisions.
4. Derive the TPU region from its zone. Require source checkpoint, dataset,
   output mirror, and any intermediate GCS bucket to be in that region. A
   one-time control-host-to-VM source copy is allowed; all workload reads must
   then be node-local.
5. Create the immutable run snapshot before every launch. Copy that exact
   snapshot to `/home/<remote-user>/elf-runs/source/<run-id>` on every worker.
   Keep downloaded `checkpoints/` out of this code snapshot; localize and
   verify checkpoints separately so ignored multi-gigabyte artifacts are not
   replicated as source files.
   Hash the selected config, sampling config, exact-restore code, and changed
   scientific source; record those hashes in the experiment report.
6. Initialize a missing runtime only on the selected exact authorized TPU.
   Treat runtime initialization, data localization, source staging, and launch
   as distinct gates so a failure cannot masquerade as a started experiment.
   `stage_elf_to_tpu.sh` must initialize before source verification only after
   every-worker probing explicitly reports `ELF_RUNTIME_MISSING`; an SSH or
   malformed probe is unknown and must fail without initialization. Set
   `ELF_TPU_INITIALIZE_MISSING=0` when staging must remain read-only. After an
   automatic initialization, rerun the complete readiness probe and require the
   marker, CPU imports, Muon, and W&B authentication on every worker.
7. Reuse a verified node-local dataset and its derived split caches. Otherwise
   copy from the region-matched bucket once, then verify `state.json`,
   `dataset_info.json`, every expected final Arrow filename, the canonical
   object and byte counts, zero `*.gstmp` transfer remnants, and free disk
   space. Deterministic derived `cache-*.arrow` files may exist outside the
   canonical source count and must not force a recopy. Do not redownload either
   canonical data or verified split caches for a second sequential evaluation
   on the same worker.
8. Immediately before fan-out, require `READY`, `HEALTHY`, the fresh output
   path to be absent, no trainer owned by the current user, every `/dev/vfio/*`
   resource idle, and no stale `/tmp/libtpu_lockfile`. A prior check does not
   satisfy this final gate. Make this a dedicated read-only occupancy check;
   never reuse a setup preflight that can initialize, delete, move, or repair
   state in the final launch race window.
9. Keep the foreground staging/launch session until it returns. Then require
   checkpoint restoration, W&B initialization, JAX distributed initialization,
   and the first useful progress event before calling the run launched.
   Before `jax.distributed.initialize()`, never call `log_for_0()` or any
   helper backed by `jax.process_index()`, `jax.devices()`, array creation, or
   device transfer. Use the launcher-provided worker index for pre-JAX messages,
   and parse retry identity overrides such as `wandb_run_id=None` as actual
   null values rather than the literal string `None`.
   When W&B starts on launcher-designated physical worker zero, keep that same
   physical worker as the post-JAX application-log, progress, validation-artifact,
   and W&B metrics owner even if automatic JAX process zero maps elsewhere. Keep
   scientific sharding and RNG ownership on the JAX process index.
10. Install one recurring liveness monitor and separately audit scientific
    artifacts. Continue until completion or explicit user stop.

## Select checkpoints from retained reality

- Enumerate the actual durable checkpoint objects and verify their byte counts;
  never infer availability from a configured save cadence, a report filename,
  or an older generation directory.
- Use the highest exact common checkpoint only when both files still exist and
  pass integrity checks. If retention removed it, say that no exact common
  checkpoint is launchable. Choose the nearest retained pair only when that
  still answers the question, and record the checkpoint-step and data-exposure
  mismatch.
- A checkpoint step is a training-loop/checkpoint step unless the code proves
  otherwise. For ELF accumulation, approximate optimizer updates as
  `training_loop_steps / grad_accum_steps` and examples exposed as
  `training_loop_steps * global_batch_size`. Do not label raw checkpoint-step
  differences as optimizer-step differences.
- Bind evaluation to one node-local regular file named `checkpoint_<step>`.
  Verify the trusted regional source URI, byte count, SHA-256, read-only mode,
  filename step, and serialized payload step. Exact evaluation must fail rather
  than fall back to a nearby checkpoint.
- Diff every parameter-bearing and source-preprocessing field against the
  checkpoint's resolved training configuration before restore: model family,
  attention heads, bottleneck and prefix-token dimensions, encoder identity and
  `train_encoder`, layer mixing, latent normalization, and any other field that
  changes parameter structure or latent coordinates. Permit an evaluation-only
  backend substitution only after proving it is parameter-free, and record the
  numerical-compatibility caveat. Keep the checkpoint's resolved config as the
  compatibility manifest; if any material field or the parameter-free status
  of an override cannot be proven, fail closed before restore.
- Use a fresh output directory and fresh W&B identity with `resume: null`,
  `wandb_resume: never`, and checkpoint saving/mirroring disabled. Restore the
  exact file before forming training-only pmaps, creating a train iterator, or
  starting a profiler; return after evaluation.

## Preserve paired-ablation validity

- Change one intervention per arm. For recurrent self-conditioning feedback,
  `alpha=0` must scale only the prior denoised prediction carried into the next
  sampler step. It must not erase the current latent, conditional prefix, or
  SC-CFG/model-mode conditioning tokens.
- Reuse identical batch seeds, initial noise, timesteps, and per-step SDE keys
  across control and treatment. Iterate the intervention inside the sampling
  parameter loop without advancing the base RNG between arms.
- Give every non-default arm a collision-proof artifact and W&B namespace. Keep
  the historical namespace only for the default arm when backward compatibility
  matters. Within one paired evaluation this means distinct metric keys and
  artifact directories inside one W&B run, not separate W&B runs, unless the
  experiment design explicitly chooses otherwise.
- Validate both arms independently: exact sample count, decoded-text emptiness
  rate, token-length distribution, EOS behavior, within-sequence repeated
  n-grams, pooled repetition, entropy, decoder/token perplexity when available,
  reference-model perplexity, and MAUVE. An exact-count all-empty or
  whitespace-only arm is a completed scientific collapse, not a reason to
  recover or rerun; classify it explicitly and record any skipped PPL or MAUVE
  as undefined. A single latest artifact is not proof that a paired evaluation
  completed.
- MAUVE is CPU-bound in the current path. In a paired sweep, the reference
  corpus, featurizer model, maximum length, and seed are unchanged, yet a naive
  per-arm call featurizes the same references again. Cache immutable reference
  features under a key containing the reference-content digest, model revision,
  maximum length, seed, and library version; compute only treatment features per
  arm. Prove cached and uncached scores match in a focused test before enabling
  this optimization. Never change a running evaluation mid-arm.
- The current sequential arm loop scores control before generating treatment,
  leaving the second arm unnecessarily exposed to Spot loss while the TPU is
  idle during CPU PPL/MAUVE. A future two-phase path should first generate and
  durably record every paired arm, then score all arms while reusing one PPL
  model and cached reference features. Prove output and metric parity in
  focused tests before enabling it; do not alter a running evaluation mid-arm.
- Parameter-free kernel substitution can preserve checkpoint compatibility but
  may introduce floating-point differences. Treat same-launch paired arms as
  controlled; do not present comparison with an older backend's generations as
  bitwise controlled.

## Hardware and compilation lessons

- TPU v4 has 16 MiB VMEM per core. The confirmed fused-Pallas ELF-B geometry
  with `pallas_block_q=1024` and `pallas_block_kv=256` uses about 17.56 MiB and
  fails permanently at compile time. For an evaluation whose parameters are
  backend-independent, use non-fused `pallas_flash` on v4 and record this as
  compatibility plumbing. For training or a kernel study, do not silently
  change the backend; use compatible hardware or a separately approved config.
- When every worker has one intended libtpu client but multi-host initialization
  reports a different launch ID, prefer a single-host replacement when it
  preserves the scientific batch geometry. More chips do not help a workload
  that is not sharded across hosts.
- Compare all-worker runtime signatures only after canonicalizing SSH transport
  carriage returns and trailing whitespace. Keep the Python, JAX, JAXLIB, and
  libtpu digest payload exact; visually identical records must not become a
  false mismatch, and genuinely different payloads must still fail closed.
- A v4-32 can pass checkpoint restore and several training-loop steps before
  failing this way. Treat the controller's `unexpected peer ... different launch
  id` / `program continuator has halted` pair as the same deterministic
  multi-host failure; the peers' later controller-disconnect errors are only
  consequences. Do not retry that topology unchanged.
- A bounded foreign-holder guard must use a wall-clock deadline, not Bash's
  inherited `SECONDS` counter; otherwise a new guard can expire immediately
  and allow the exact external controller to reclaim VFIO during staging.
- Tokenize `fuser` output as whitespace-separated numeric PIDs before a bounded
  foreign-holder guard validates or signals it. One process can appear once per
  VFIO resource and the command may omit its final newline; line-oriented
  readers can therefore silently skip the holder while the guard remains live.
- Never infer useful throughput from steps/second across different global
  microbatches. Use
  `(training_loop_steps * global_batch_size) / elapsed_seconds`, excluding
  compile, evaluation, checkpointing, profiling, and warmup.
- For a proven fixed-prefix continuation crop, project decoder vocabulary logits
  only for supervised suffix positions. Gate this with an explicit start
  position that exactly matches the fixed condition length, preserve full token
  labels and masks for every other objective, and prove suffix loss/feature
  parity before benchmarking. Never infer a fixed prefix from a maximum length.

## Integrity and dependency failure memory

### Legacy Cloud SDK can list and copy but not rsync

Probe `gcloud storage ls`, `rsync`, and `cp` independently. Some SDK `428.0.0`
images can list and copy authenticated regional objects but do not expose
`gcloud storage rsync`, while their `gsutil` fails during the OpenSSL import.
Use the working `gcloud storage ls` result for the trusted remote byte total;
prefer `rsync`, then use `gcloud storage cp --recursive '<prefix>/*' '<local>/'`
when only `cp` is available. Require dataset metadata, exact bytes, and zero
temporary files on every worker. Check exact local reuse before invoking any
broken `gsutil` fallback.

### CPU control-host tests import Torch-XLA and segfault

Transformers or Accelerate can transitively import `torch_xla` during pytest
collection even when JAX is configured for CPU. A native crash with no Python
traceback is not a failed model test. Rerun CPU-only tests with both
`USE_TORCH_XLA=0` and `JAX_PLATFORMS=cpu`, preserve the first crash log, and
require the focused suite to pass under the explicit CPU environment.

### Composite GCS object requires compiled CRC32C

Older `gsutil` can refuse a composite-object download when compiled `crcmod` is
missing. Download into a fresh temporary path on the same destination
filesystem as `checkpoint_<step>` with
`-o GSUtil:check_hashes=never` only when an independently trusted byte count and
SHA-256 are already recorded. Verify both immediately after transfer, set the
file read-only, and atomically promote it to `checkpoint_<step>`. Never weaken
hash checking without the independent end-to-end digest, never overwrite a
verified destination, and never accept a partial file.

The trusted digest must predate and be independent of the disabled-hash
transfer—for example, a recorded SHA-256 from a previously verified durable
copy or immutable experiment manifest—and the report must also pin the GCS URI,
object generation, and byte count. A digest computed only from the newly
downloaded no-hash file is not independent provenance.

For large dataset `rsync`, missing compiled `crcmod` also creates a long
single-core verification tail and disables sliced downloads. Install and
preflight compiled `crcmod` during runtime preparation when possible, or use an
already verified node-local dataset. Do not interrupt a nearly complete first
copy merely to optimize the verifier; subsequent sequential runs should reuse
the verified local snapshot.

During a first-use legacy-`gsutil` promotion, live transfer/checksum workers and
decreasing temporary-object state mean active `STAGING`, even when W&B and JAX
have not started. An SSH timeout during that state is not evidence of a stall,
trainer failure, or preemption. Reconnect and inspect the same exact transfer;
never start an overlapping retry while it remains live.

Declare an abandoned transfer only after no matching transfer or checksum
worker is live and neither temporary bytes nor modification time advances over
two direct checks at least 60 seconds apart. Preserve the dead attempt and its
temporary path in the report, then use a distinct fresh temporary path; never
overwrite or promote the abandoned partial file.

### Training Python and Cloud SDK Python diverge

Treat them as separate compatibility domains. Never blindly set
`CLOUDSDK_PYTHON` to `ELF_RUNTIME_PYTHON`. Preflight the exact Cloud SDK transfer
command against a harmless run-scoped regional object. Record the interpreter,
SDK version, command, and outcome.

### W&B opens libtpu or receives a stale identity

Create a fresh deterministic evaluation display name before JAX, set
`wandb_run_id: null` and `wandb_resume: never`, start W&B only on worker zero,
temporarily constrain its subprocesses to CPU, and disable system statistics.
W&B assigns the run ID; after startup, read the actual ID/link from the worker
log or API and never derive it from the display name. A successful `wandb.init`
is not proof that checkpoint restore or generation began.

## Exact-evaluation monitoring

The current `watch_elf_run.sh` is a safe infrastructure/liveness watcher for an
evaluation-only run with `checkpoint_gcs_path: null` when its final workload
role is `evaluation`,
`ELF_MONITOR_EVAL_INTERVAL_STEPS=0` and
`ELF_MONITOR_EXPECTED_GENERATION_SAMPLES=0`. Pin
`ELF_MONITOR_TRIGGER_CODEX=false` so this liveness-only wrapper cannot start
automatic recovery, set `ELF_MONITOR_CHECKPOINT_MIRROR_GRACE_SECONDS=0` because
no mirror is expected, and set
`ELF_MONITOR_IGNORE_TERMINAL_WANDB_WHILE_TRAINING=true` to tolerate the brief
process/W&B terminal-state race. It is not a scientific completion validator
for paired exact-checkpoint evaluation:

- its step parser does not recognize every exact-restore message;
- scheduled-evaluation logic rounds to training cadence boundaries;
- it examines only the latest generation artifact, not every intervention arm;
- process exit can briefly precede the W&B terminal-state update;
- a long silent evaluation can look like a stale training log.

For paired exact evaluation, disable automatic recovery in that generic watcher
and keep recovery decisions in one active orchestrator. At each check, directly
verify the exact TPU state, trainer state, fresh log progress, W&B state, and
both expected artifact namespaces. Completion requires process exit, W&B
`finished`, the exact restored step, validation output, exactly the requested
sample count in every arm, and either complete per-arm generation metrics or an
explicit collapse classification with inapplicable PPL/MAUVE recorded as
undefined. Emptiness alone must never trigger recovery. Treat the short
process-exit/W&B-state race as pending, not failure.

The W&B/process race allowance is bounded by the next direct orchestrator check
and never by automatic recovery. If W&B is terminal while an evaluation process
remains live at the next check, classify and alert on the W&B/logging failure;
do not let the liveness wrapper silently mask it for another interval.

Distinguish infrastructure loss from application failure:

- Exact API `NOT_FOUND`, `PREEMPTED`, `DELETING`, or an unhealthy/unusable node
  is TPU loss.
- A live usable TPU with a missing trainer is an application exit; diagnose its
  log before any relaunch.
- Authentication, DNS, SSH, missing executable, or W&B API failure is monitor
  infrastructure failure, not evidence of preemption.
- W&B `crashed` alone never proves Spot preemption.

At normal training completion, require any reported trainer to have a live
non-zombie `/proc` entry, an exact `train.py` argv token, and the exact run
identity. A remote inspection shell can contain the same text and must not
count as the trainer. When W&B is `finished`, the configured limit and final
evaluation are logged, and the terminal checkpoint marker matches the local
and regional byte count plus SHA-256, terminal trainer or mirror zombies are
completed children awaiting foreground-wrapper reaping, not recovery evidence.
Record completion and never launch a retry from that state.

Do not replace the log worker's shell with `tail --pid`: a completed trainer
can remain as that tail's zombie child, keeping the tail and foreground fan-out
alive indefinitely. Run the log tail as a separate child under exit and signal
cleanup traps, deferring signal exit until the tail PID has been captured so a
fork-to-publication signal cannot orphan it. Start the trainer beneath a
detached node-local supervisor that owns and reaps it and atomically persists
its PID, readiness, and terminal status independently of the SSH-bound shell.
Acquire an atomic per-worker
in-progress claim before fallible launch preparation and remove it only on a
proven pre-spawn exit. After spawning may have begun and the supervisor,
trainer, and required checkpoint mirror PIDs are atomically published, promote
the claim to a distinct persistent committed marker. Retain the pre-commit
signal traps across that promotion until the post-commit handlers replace
them; never expose a visible committed marker during default signal handling.
After live-tail cleanup, replace the cleanup handlers directly with
deterministic terminal handlers until status reporting exits; do not restore
defaults while terminal work remains. An SSH retry returns a stored terminal
status, succeeds only while the exact complete argument vectors
of both supervisor and trainer are verified live, and otherwise fails
indeterminate without another spawn. After readiness, poll across the legal
fast-exit boundary until either those live identities validate or the atomic
terminal status is present; do not strand a completed attempt as permanently
in-progress merely because the trainer exited between the two observations.
The checkpoint mirror must also recognize that atomic status, perform its final
scan, and exit without waiting for the startup timeout when a short trainer
ended before its first live-process sample. While the run is active, use the
atomically published supervisor PID, process-generation start time, and
readiness evidence; a fresh trainer argv
need not contain the configured run ID, so a fuzzy run-ID `pgrep` can time out
during healthy training. The v5e-16
b128/BKV512 MFU gate exposed this failure after a
successful step-800 exit: terminating the stale tail caused `gcloud` to retry
the remote command, and only W&B `resume: never` prevented a duplicate trainer.

When a watcher starts persistent recovery under `nohup`, close its flock file
descriptor in the child. Otherwise the recovery process owns the monitor lock
for its full session and every scheduled check is skipped. Keep the inherited
recovery as the sole orchestrator, apply the close-on-spawn fix, and let the
installed monitor resume after that exact recovery exits.

Shell detachment is also not systemd-cgroup isolation. Before retiring an
obsolete precursor unit, compare `/proc/<pid>/cgroup` for the precursor
recovery and every successor monitor. Never stop the precursor while the
authoritative successor monitor or current recovery remains in its cgroup.
First revalidate and TERM only duplicate host-monitor groups, then install one
TERM-safe successor monitor in a dedicated unit and require a fresh healthy
check with exactly one cadence-sleep child. If the recovery itself still lives
in the precursor cgroup, use one exact-identity cleanup helper in a third cgroup
that waits for the recovery to exit, revalidates both unit commands, and retires
only the obsolete precursor. Never signal the remote trainer during this
monitor migration.

On a multi-worker topology, do not assume that automatic JAX process zero and
the launcher-designated checkpoint mirror worker are the same physical VM.
Pin them independently: read training progress, evaluation artifacts, and the
newest checkpoint candidate from the process-zero log worker; read the
canonical mirror PID, exact helper identity/root count, and completion marker
from the mirror worker. A missing mirror PID on the log worker is not a helper
failure when the separately verified mirror worker has the one live helper and
matching durable checkpoint evidence. Keep that helper running instead of
moving or duplicating it to satisfy the monitor.

On confirmed Spot loss, reconcile the full report/W&B/monitor lineage first,
select the newest verified durable checkpoint, retain exactly one recovery
orchestrator, obtain replacement capacity from an authorized `kristine` TPU, or
from an idle non-`kristine` TPU only after the ownership key phrases, recreate
the same immutable source/config/region-local data, and resume without changing
the scientific intervention.

If a paired evaluation fails or is preempted before both arms are durably
complete, use a fresh retry output path, W&B identity, immutable snapshot, and
report, then rerun the complete pair from the same exact verified checkpoint.
The retry snapshot must contain the precursor's byte-identical source and
effective configuration under a new retry identity, not a new capture of the
mutable worktree. Never append a treatment arm to a precursor attempt's partial
control arm. If the exact compatible checkpoint cannot be verified, stop; an
exact evaluation may not substitute another step or restart from step zero.

## Reproducible reporting and artifact handoff

### Read one addressable replica when saving JAX 0.11 pmap state

JAX 0.11 pmap state can expose a globally sharded leading replica axis. Do not
unreplicate multihost checkpoint leaves with `x[0]`: that integer index can
compile a cross-host gather/reshard and fail limited-ICI device assignment at
the checkpoint boundary. For replicated training state, read
`array.addressable_shards[0].data.squeeze(0)` on each process before
`jax.device_get`, then serialize the process-local replica. Keep a focused test
that rejects `jax.Array.__getitem__`, and require the first scheduled checkpoint
to complete on the exact multihost runtime before treating checkpointing as
proven. If process zero fails while another worker writes and mirrors the file,
accept that checkpoint only after marker, bytes, SHA-256, and payload-step
verification; the existence of a same-named file alone is insufficient.

### Short claim windows benefit from verified prelocalization

When a large region-matched dataset is already verified in the selected
regional `kristine` bucket but `/home` cannot hold it, an isolated
`/dev/shm/elf-*` destination is valid node-local storage. Before claiming the
accelerator, synchronize while leaving VFIO untouched, then require the
dataset's completion metadata, zero temporary transfer files, and exact source
and destination byte counts. Stage the immutable source snapshot at the same
time. This narrows the guarded claim-to-backend-open window to the final
identity checks and launcher handoff; it does not relax the final occupancy
gate or authorize cross-region dataset transfer.

Validate every W&B tag before staging or claiming capacity. W&B accepts at
most 64 characters for a tag; a longer tag is a deterministic zero-step
preflight failure and requires a fresh run/output/W&B identity after shortening
the tag.

### v5e fused-Pallas VMEM differs from the non-fused optimum

On v5e with ELF-B, length 1024, three 256-dimensional heads, and four
examples/chip, do not assume the v6e fused-Pallas winner will compile. A
`pallas_flash_fused` `BQ1024/BKV256` model-init call requested `17.14 MiB`
against v5e's `16.00 MiB` scoped-VMEM limit. Reducing `BQ` to `512` cleared
model initialization, but the full training backward still returned per-batch
Q/K/V gradients through a `23.39 MiB` custom-call allocation. Treat those as
deterministic zero-step compatibility failures, not HBM or TPU-health failures.

Use a fresh identity to gate the non-fused `pallas_flash` boundary before
shrinking the scientific batch. At the same geometry, `BQ1024` compiled with
`BKV128`, `256`, `512`, and `1024`; the measured clean means were respectively
`188.0583`, `189.69225`, `191.726995`, and `191.46773 examples/s`. Select
`BQ1024/BKV512` for this v5e workload: it reached `30.923%` matched-work modeled
MFU and a `76.398704 ms` representative train-step execution. Re-gate after
changing per-device batch, head geometry, sequence length, JAX/libtpu, or v5e
topology; the tile result is not universal.

- Create the experiment report before launch with status `planned`, the exact
  command, immutable snapshot, checkpoint integrity, seed, baseline/treatment,
  changed fields, expected namespaces, and verification booleans.
- The YAMLs are complete prelaunch inputs, not necessarily the resolved runtime
  configuration. ELF derives values such as microbatch and learning rate and
  may derive the evaluation W&B display name. After startup, copy the TPU-written
  `output/config.yml` into the report verbatim as the canonical complete
  effective configuration and record its SHA-256.
- For an exact-checkpoint evaluation, resolve the canonical source training run
  from checkpoint ancestry and set `wandb_group` to that exact source run ID.
  Reuse the same group across checkpoint steps, sampling policies, retries, and
  hardware while keeping every evaluation's W&B run ID and display name unique.
  Verify the effective config before launch and the created W&B run after
  startup, then record the group in the experiment report. Never mutate older
  W&B runs without an explicit user request naming the intended run set.
- Append the actual start/finish times, W&B ID/link, checkpoint restore line,
  progress transitions, failures/recoveries, metrics, and artifact paths. Never
  overwrite a prior attempt or reuse its output/W&B identity.
- Pull generations and metric JSONL files to a unique run-scoped directory on
  the shared mount after completion. Verify byte counts, line counts, hashes,
  and arm-to-namespace mapping before comparing results or presenting samples.
- For long-running training, keep monitoring after launch and append material
  status changes. For sequential evaluations on one TPU, start the second only
  after the first process is gone, W&B and artifacts are terminal, and VFIO is
  idle again.
- After a terminal evaluation, remove only that run's cron entry, wrapper, and
  recovery trigger. Before starting a sequential successor, require its
  predecessor process gone, W&B/artifacts terminal, shared-mount outputs copied
  and hashed, and VFIO idle; preserve every unrelated installed monitor.

---
name: query-monitor-gcp-tpus
description: Safely find, provision, inspect, and claim Google Cloud TPU VMs for ELF. Use for TPU capacity queries, tpu_scripts/query_tpus.sh, READY checks, VFIO occupancy checks, TPU selection before launch, Spot capacity failures, or replacement capacity. Always prefer resources whose names contain kristine. A wide read-only sweep of other TPUs, and later acquisition of an idle one, require the exact key phrases in AGENTS.md TPU Ownership Safeguard. Never interrupt, affect, or announce yourself to another person's run.
---

# Query and Claim GCP TPUs

## Keep these barriers

- Always prefer resources discovered with a server-side `kristine` name filter. On the default path, reject every describe, SSH, process action, or other TPU call unless every exact TPU ID contains `kristine`, case-insensitively. The only exception is an explicit current-message key phrase from AGENTS.md TPU Ownership Safeguard: `AUTHORIZE WIDE READ-ONLY TPU SWEEP` permits a read-only sweep of other TPUs solely to determine that they are not actively held. During that sweep, never interrupt a run, affect a run, or make yourself known to a run. Once a read shows an active holder, end that visit without touching the job and continue the rest of the sweep. A later full sweep may read-only re-query a TPU previously found actively held, because its job may have finished and the TPU may be inactive again. Every sweep stays read-only and must never touch an existing active job. On a non-`kristine` TPU, do not edit, stop, or touch an actively held run owned by somebody else at any point, even when the user requests it. A process owned by kristine may be terminated only when the user explicitly authorizes that termination, and never as part of a sweep. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it. On a `kristine` TPU, the documented recovery rules apply. `AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION` plus the exact TPU name is required before acquiring a TPU the read showed is not actively held.
- Never reboot, reset, restart, stop-and-start, or reimage a TPU or its guest OS. If the kernel is stuck, leave the TPU untouched and choose another authorized `kristine` TPU.
- A healthy `kristine` workload remains hands-off. Standing user authorization permits ending a current-user ELF process only when it is verified stuck or race-blocking the authorized run: prove the exact `kristine` TPU, owner, PID, full command, and run identity; record lack of progress; send TERM; revalidate before KILL escalation; and confirm all workers' VFIO resources become idle. Never use this exception for foreign, unknown, non-ELF, or healthy processes. On a `kristine` TPU, the occupancy helper may otherwise terminate only a non-`kristine`, foreign process that currently holds `/dev/vfio/*`. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it.
- Standing user authorization permits clearing a holderless `/tmp/libtpu_lockfile` without asking again, but only on an exact `kristine` TPU. Immediately before removal, successfully inspect every worker and prove that no PID holds any `/dev/vfio/*` device or the lockfile, no `train.py` process is active, and no worker is `UNKNOWN`; the stale lockfile must be the only blocker. Recheck those invariants inside the cleanup command, remove only `/tmp/libtpu_lockfile` on workers where it exists, then rerun the occupancy helper and require `AVAILABLE`. This authorization never covers another file, process termination, an occupied or unknown worker, or a TPU whose name lacks `kristine`.

## Find existing capacity; provision only on explicit request

Run the reconciler in read-only discovery mode. It applies the server-side owner filter, checks current state, health, occupancy, and runtime readiness, and prints workload-agnostic candidates in its final `AVAILABLE` table without initializing missing runtimes:

```bash
ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group
```

The checker streams one row after each TPU finishes its probes. Never select
from partial output: wait for the command to exit successfully, then parse only
the final `AVAILABLE` section. If the user supplies an exact `kristine`
`NAME,ZONE` pair that Cloud Asset Search has not indexed yet, pass it through
`ELF_TPU_EXACT_CANDIDATES`; the checker will apply the same state, health,
occupancy, and runtime probes before including it in the final table.

Select candidates only from the final `AVAILABLE` table. Do not treat `ELF_RUNTIME_STATUS=MISSING` or `UNKNOWN` as launch-ready; initialize and revalidate only the one selected idle TPU. Do not use `scripts/sweep_tpus.sh` for new discovery workflows.

If no suitable ID is returned, do not ask the user for a TPU name. Keep rerunning
the same filtered `check_tpus.sh` audit on the monitoring cadence. For an active
v6e run recovering from confirmed TPU loss, an existing `AVAILABLE` v5p-8,
v5p-16, or v5p-32 is an approved fallback when no matching v6e is available.
Keep the same active-run count and Spot policy; require the checkpoint, dataset,
and future outputs to remain in the replacement's region. Treat it as a new
retry and reapply compatibility, HBM, and throughput gates before launch.

Never invoke `tpu_scripts/query_tpus.sh` unless the user explicitly requests new
TPU provisioning in the current conversation. A generic launch, resume,
relocation, recovery, or persistence request does not authorize provisioning.
Only after that explicit request, route the requested accelerator type and
topology to:

```bash
bash tpu_scripts/query_tpus.sh TYPE SIZE 8
```

Allow these type and size combinations:

- `v5e`: `8`, `16`, `32`, or `64`.
- `v5p` and `v4`: `8`, `16`, or `32`.
- `v6e`: `8`, `16`, `32`, `64`, `128`, or `256`.

The script restricts creation attempts to GCP-surveyed zones with an advertised
shape, positive effective Spot quota, and a matching regional bucket service
account: v5e uses `asia-northeast1-b`, `us-west4-a`, or `us-central1-a`; v5p
uses `us-central1-a` or `us-east5-a`; v6e uses `us-central1-b`,
`us-central1-a`, `us-east5-a`, `asia-northeast1-b`, `us-east5-b`, or
`us-central1-c`; and v4 uses only `us-central2-b`. An explicit
`ZONES_OVERRIDE` must be a subset of that family allowlist or the script exits
before any cloud request. Use the type and size required by the run, as well as
any above it that will allow the run.

The 2026-09-23 read-only survey established these slice-to-machine counts:

| requested type | TPU VMs per requested TPU |
| --- | ---: |
| v5e-8 / 16 / 32 / 64 | 1 / 2 / 4 / 8 |
| v5p-8 / 16 / 32 | 1 / 2 / 4 |
| v6e-8 / 16 / 32 / 64 / 128 / 256 | 2 / 4 / 8 / 16 / 32 / 64 |
| v4-8 / 16 / 32 | 1 / 2 / 4 |

The survey used accelerator-type and topology metadata, effective per-zone
quota limits, aggregate Cloud Monitoring quota usage, bucket locations, and
`testIamPermissions`. It did not create or queue capacity and did not inspect a
non-`kristine` TPU. GCP does not expose live Spot physical capacity through a
read-only API, so a cataloged zone can still return `RESOURCE_EXHAUSTED`.


## Wide read-only sweep of other TPUs

Prefer the filtered `kristine` inventory above. A sweep of every other TPU is a separate workflow, and only when the user's current message contains the exact phrase `AUTHORIZE WIDE READ-ONLY TPU SWEEP`. A launch, resume, relocate, recovery, or capacity request does not grant it.

1. Finish the `kristine` `check_tpus.sh` inventory first and use a suitable `kristine` candidate when one exists.
2. List and describe other TPU resources read-only: cloud state and health only. Do not call `scripts/sweep_tpus.sh`, `check_tpu_occupancy.sh`, or `tpu_scripts/query_tpus.sh` for this sweep. Those tools stay on the `kristine` path; `sweep_tpus.sh` refuses other names, and `check_tpu_occupancy.sh --claim` terminates holders.
3. A guest read is allowed only to decide whether the TPU is actively held. Use one non-interactive read of process and VFIO occupancy. Do not write, signal, lock, copy, reboot, reset, reimage, start a workload, or leave a session connected.
4. Reading another person's TPU is possible under that phrase. It is never acceptable to interrupt a run, affect a run, or make yourself known to a run. When the read shows an active holder, end that visit: do not announce it or touch the job's processes, logs, checkpoints, or devices, then continue the sweep. On a non-`kristine` TPU, do not edit, stop, or touch an actively held run owned by somebody else at any point, even when the user requests it. A process owned by kristine may be terminated only when the user explicitly authorizes that termination, and never as part of a sweep. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it. On a `kristine` TPU, the documented recovery rules apply. No later message or key phrase waives the ban on touching somebody else's active job.
5. Record both actively held and inactive TPUs. Leave every active job untouched.
6. Include every TPU previously recorded as actively held in the next full sweep. A later full sweep may read-only re-query a TPU previously found actively held, because its job may have finished and the TPU may be inactive again. Every sweep stays read-only and must never touch an existing active job. If the new read still shows an active job, record that and leave the job untouched.
7. Acquire one only after a later message contains `AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION` together with that TPU's exact resource name, and only when the latest read-only check showed it is not actively held.

## Focus on the selected TPU ID

Once an ID is found, stop broad discovery. Validate its exact name, then describe only that ID to confirm its current state, health, type, topology, zone, and Spot/preemptible policy. On a non-`kristine` TPU, do this only after both key phrases above, and only for the exact idle TPU the user named.

Before launch, inspect VFIO occupancy. This helper accepts only a `kristine` TPU. For an acquired idle non-`kristine` TPU, repeat the wide-sweep non-interactive occupancy read immediately before launch, and abort if that TPU is now actively held:

```bash
bash .agents/skills/query-monitor-gcp-tpus/scripts/check_tpu_occupancy.sh \
    he-vision-group TPU_ID ZONE
```

- `AVAILABLE`: use the TPU.
- `OCCUPIED`: a `kristine` workload or current-user process owns VFIO; leave it alone unless it meets the verified stuck/race recovery exception above. If the only evidence is a holderless `reason=stale_libtpu_lock`, apply the standing stale-lock authorization above and recheck. Otherwise select another `AVAILABLE` TPU or rerun `check_tpus.sh` later.
- `RECLAIMABLE`: the holders are foreign and non-`kristine` on this `kristine` TPU. Run the same command with `--claim` immediately before launch; it terminates only those revalidated VFIO holders and never reboots the TPU. Never use `--claim` during a wide sweep, and never use it on a TPU whose name lacks `kristine`. If a run is owned by somebody else on a non-`kristine` TPU, never edit, stop, touch, or interrupt it.
- `UNKNOWN`: do not touch the TPU.

For any occupancy result, record owner, command, and run identity. If a stale lock is cleared, record the exact TPU, workers, lock path, pre-removal evidence, and post-removal occupancy result. If the holder is the active selected ELF run, leave it running and monitor it rather than launching a duplicate.

After a claim, rerun the check and launch only when it reports `AVAILABLE`.

## Report

Report the exact TPU ID, zone, type/topology, state, health, Spot flag, occupancy result, and any capacity failure. Prove readiness from current API state rather than query-script output alone.

# Qwen Island Community comparison

Status: implementation and remote validation underway; no completed pretrained-model experiment yet.
User requested ten initial islands and frozen Qwen3-0.6B with history, then a
three-experiment comparison including independently reward-trained Qwen adapters.
A follow-up is pending to distinguish the two learning arms. Proposed arms:
frozen; per-island immediate-reward adapters; per-island adapters with TD critic.
Only the frozen baseline is selected for its first experiment at this point.

Source commit: ad2ab02d789e7a12c2f1d1cd42b89f5e9a55a805, private
https://github.com/kristinelu921/island-community .
Source archive SHA-256:
`ec136cec09c7ad5c61b3fb50d8f3205b58b74f42e5c0470bc1be62190954a59e`.
Primary config: `island_community/scenarios/qwen_10_frozen.json`.
Seed 42; ten founders; five mutual neighbors with grain/water access; eight
private history turns. Producers: water 2, grain 2, timber/stone/iron/clay/fiber/herbs
one each; salt/fuel zero. The ten-island preset explicitly permits those missing
producer types; previous presets remain strict. Health 300, original rewards,
all feasible integer trade quantities, automatic births at interval 20.

Model: Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca.
Frozen shared float32 weights, no optimizer or updates in baseline. Thinking
budget 96 tokens per action/quantity stage, constrained final JSON choice,
temperature 0.6, top-p 0.95, top-k 20, context limit 16384 without truncation.
Full before/after base-weight hash, exact rendered prompts, reasoning, raw JSON,
menus, resolved actions, execution results, histories, rewards and rule events
are recorded. Grammar failure aborts without a substituted action.

Authorized remote CPU: worker 0 of
kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
he-vision-group / us-west4-a. Filtered check_tpus inventory AVAILABLE/READY;
fresh VFIO occupancy AVAILABLE. At most four CPU threads, no TPU initialization.
Task root `/tmp/island-community-qwen-20261110-224000`.
Isolated environment `qwen-venv`: Python 3.10.18, torch 2.6.0 CPU,
transformers 4.51.3, numpy 1.26.4, pytest 8.3.5. See dependency failure/recovery
records for the preserved ensurepip setup failures; no model ran in those attempts.

Validation so far: 126 existing local rule/storage/neighbor tests; 21 new local
language-contract tests; six local HTTP/viewer tests; four remote neural tests
using a small actual Qwen architecture and the real pinned tokenizer. The neural
tests cover constrained decoding, trainable-island isolation, unchanged base
weights, inherited adapters with fresh optimizers, checkpoint resume, and full
runner action recording. Full committed-source remote suite is in progress.

Selected baseline command, from task root/source-v1:

```bash
HF_HOME=/tmp/island-community-qwen-20261110-224000/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/time -v -o /tmp/island-community-qwen-20261110-224000/frozen-timing.txt /tmp/island-community-qwen-20261110-224000/qwen-venv/bin/python -m island_community run --config island_community/scenarios/qwen_10_frozen.json --turns 20 --device cpu --checkpoint-every 1 --output /tmp/island-community-qwen-20261110-224000/frozen-recording
```

This is a bounded initial comparison, not a convergence or statistical superiority claim.

## Committed-source validation and baseline launch

All 316 remote tests passed in 293.26 seconds with the pinned isolated runtime.
The full log is node-local `full-tests-v1.log`. Fresh occupancy immediately before
launch was AVAILABLE on worker t1v-n-4b587646-w-0. The immutable source archive
checksum matched after remote copy. The duplicate-guarded detached launcher is
preserved as `launch-frozen.py`; it uses the exact baseline command above and
writes frozen.pid, frozen.log, and frozen-timing.txt. The pretrained-model
baseline is selected for 20 turns with one checkpoint per turn; no learning-arm
experiment has been launched while the third-arm distinction is pending.

Baseline launch verified: timing-wrapper PID 3556655 and Python PID 3556656.
The first process check showed active CPU execution and about 3.8 GiB RSS.
No TPU device was used.

## Initial pretrained CPU attempt preserved

The first decision was still computing after 189 seconds (313% CPU, RSS
6,233,268 KiB), with only the founding checkpoint at turn 0. The source repeated
the full long action prompt for final constrained decoding. This is a measured
latency concern, not a formatting failure or stalled process.

The exact current-user Python PID 3556656 was verified by owner and argv, then
sent SIGINT to stop this task's own run for a prompt/cache optimization. The
original frozen-recording, log, timing, and checkpoint remain untouched.
No economic result or completed turn is claimed for this attempt.

Selected repair under verification: group action-menu rows by kind to remove
repeated field names while keeping every original menu ID, and carry Qwen's
attention cache from reasoning into constrained final-answer generation. These
changes affect prompt formatting and inference work, not the world rules or
requested-action mapping. A fresh output identity will be used for the retry.
The user has also been asked whether the earlier four-CPU-thread limit may be
raised to 32; the four-thread limit remains in effect pending that answer.

The interrupted initial attempt consumed 6:04.38 wall time, 1143.05 user CPU
seconds, 316% CPU, and peak RSS 6,739,828 KiB. It terminated by signal 2,
with zero completed turns. GNU time also printed an exit-status field of zero;
that field is not treated as success because the signal, KeyboardInterrupt,
manifest, and turn-zero checkpoint establish interruption. The prompt/cache
change is being verified in source-check2 before any scientific retry.

## Cache/menu verification and occupancy change, 23:14 UTC

The source-check2 remote focused suite passed 25 tests in 37.08 seconds.
Reproduction command from source-check2:
`ISLAND_ALLOW_NEURAL_TESTS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/tmp/island-community-qwen-20261110-224000/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/island-community-qwen-20261110-224000/qwen-venv/bin/python -m pytest island_community/tests/test_qwen_backend.py island_community/tests/test_language_policy.py -q`.
Its log remains at `cache-tests-check2.log`. These neural tests use the small
Qwen architecture fixture with the pinned real tokenizer, not pretrained-model
economic results. An additional local configuration guard test and exhaustive
action-menu field assertions passed with all 22 local language-contract tests:
`python -m pytest island_community/tests/test_language_policy.py -q`.

The tested cache/menu changes and the all-feasible-quantity configuration guard
are committed and pushed to private GitHub main as `d119758`. The original
interrupted run remains preserved. No retry has been started.

A fresh read-only occupancy check of the same authorized VM returned OCCUPIED:
worker `t1v-n-4b587646-w-0`, current-user holder PID 3563018, owner `kristinelu`,
command `/tmp/kristine_bench_all11_promptdiffusion_20261110/env/bin/python -u /tmp/kristine_bench_all11_promptdiffusion_20261110/qualify_native8.py --candidate A1 --model /dev/shm/kristine_bench_all11_promptdiffusion_assets_20261110/assets/model --official-source /tmp/kristine_bench_all11_promptdiffusion_20261110/sources/official --launch-receipt /tmp/kristine_bench_all11_promptdiffusion_20261110/qualification_native8_launch.json --output /tmp/kristine_bench_all11_promptdiffusion_20261110/qualification_native8 --cache /tmp/kristine_bench_all11_promptdiffusion_20261110/cache`.
That separate workload was left untouched. Qwen has only used CPU on this VM;
no TPU accelerator use is claimed or attempted. The earlier four-thread limit
still applies. The requests to distinguish the two learning conditions and to
raise the CPU limit remain unanswered.

## Read-only capacity audit, 23:22 UTC

The user's request to identify the holder and other available accelerators was
checked with `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`.
The completed final AVAILABLE table contained two resources:

- `kmh-tpuvm-v4-8-spot-kristine-10-23-08-42-N-us-central2-b`
- `kmh-tpuvm-v4-8-spot-kristine-09-14-38-39-N-us-central2-b`

Fresh cloud describes confirmed both are v4-8, topology 2x2x1, READY, HEALTHY,
and Spot in us-central2-b. Both passed the VFIO occupancy audit. Their reusable
ELF runtime is MISSING; automatic initialization was disabled. These are idle
hardware candidates, not validated Qwen accelerator environments. The current
Qwen implementation supports CPU and CUDA, so TPU acceleration needs a backend
implementation and remote validation before launch.

The same audit still found PID 3563018, owner kristinelu, holding all eight
numbered VFIO devices on the previous v5e-8. Its exact candidate-A1 qualification
command is recorded above. The v6e-16 kristine resource was DELETING.
`gcloud compute instances list --project=he-vision-group --filter='name~kristine AND guestAccelerators:*' --format='json(name,zone,status,machineType,guestAccelerators)'`
returned no matching GPU VMs. No resource was provisioned, initialized, claimed,
or modified, and no workload was interrupted or launched during this audit.

## Refreshed own-pool audit following wide-sweep request

A second completed `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`
audit returned both previously listed v4-8 resources plus the original v5e-8
`kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`
in its final AVAILABLE table. The v5e-8 had ELF_RUNTIME_STATUS READY; both v4-8
resources still had MISSING. This establishes newly idle hardware, not the
completion status of the separate candidate-A1 job. No workload was launched.
The new wide sweep remains pending the exact current-message authorization
phrase required by TPU Ownership Safeguard; no non-kristine TPU was inspected.

## Authorized wide read-only sweep completed, 23:30 UTC

The user supplied `AUTHORIZE WIDE READ-ONLY TPU SWEEP` for this new inspection.
After refreshing the filtered kristine pool, Cloud Asset Search identified five
inventory zones. Current TPU list calls in those zones returned 122 resources:
68 fully idle, 13 occupied, four unresolved due to unheld libtpu lockfiles, and
37 excluded (21 creating, 13 deleting, three without confirmed healthy state).
No cloud query failed. The guest checks ran from 23:29:39 to 23:30:39 UTC.

Idle resources: 46 v5e-16 and eight v5e-32 in us-west4-a; eight v6e-16 in
us-east5-b; four v4-8 in us-central2-b; one v5e-8 and one v5e-4 in us-west4-a.
Three idle resources have kristine in their names, and 65 do not. Every worker
was inspected before a resource was called idle (286 worker checks for those
68 resources). An occupied resource was left immediately upon identifying an
active holder; no later workers of that resource were visited. Unheld locks
were left untouched and excluded from the idle set.

Exact command: `python /tmp/island-community-wide-tpu-sweep-20261110.py`.
An immutable source copy, remote read-only probe, cloud snapshots, per-resource
results, and summary JSON are preserved in `wide-sweep-20261110-2327/`.
The probe uses existing SSH credentials, fuser, and a minimal process check;
it performs no remote writes, installation, acquisition, signals, or workloads.
No accelerator use or non-kristine acquisition is authorized by this sweep.

The searchable local report is served at `http://127.0.0.1:42187/`, detached
server PID 4051668. HTTP checks confirmed all 122 resource rows, 68 idle rows,
exact summary counts, and successful checks of every worker for each idle
resource. This is a timestamped snapshot, not a live availability guarantee.

## Exact v6e acquisition authorization and native TPU setup

The user supplied the separate exact acquisition phrase for
`kmh-tpuvm-v6e-16-kaiminghe-070b9f`. Fresh cloud metadata confirmed v6e-16,
topology 4x4, READY, HEALTHY, Spot in us-east5-b. Fresh checks found all four
workers idle. Worker 0 has 180 logical CPUs, 708 GiB RAM, and initially 75 GiB
free local disk. No other resource was acquired.

Task-local root: `/tmp/island-community-qwen-tpu-20261110-234000` on worker 0.
The first dependency attempt failed because TorchTPU requires NumPy>=2.
The separate `venv-native` recovery environment installed Python 3.12.16,
torch 2.15.0+cpu, torch-tpu 0.2.0, libtpu 0.0.52, transformers 5.22.2,
numpy 2.5.4, accelerate 1.15.0, and pytest 8.3.5. Exact packages are in the
remote `dependencies-native.txt`. The first environment is preserved.

Prepared application changes add a native TPU device branch, CPU-side generation
tokens, a static attention cache, the public single-device launch helper, and
device-selectable remote neural tests. These changes are not published yet.
All 22 local action-contract checks passed. The first native neural check failed
at topology initialization before model inference (11.69 seconds); it is
preserved separately in `v6e-native-check1-failure.md` and remote native-check1.log.

The next preflight stopped before retry because worker 0 had an empty unheld
`/tmp/libtpu_lockfile`, owner kristinelu / UID 2014, modified exactly
2026-11-10 23:44:51.148260875 UTC during our failed initialization. No PID held
any VFIO resource or the lockfile at the follow-up read. The file was not removed.
Explicit cleanup permission was requested because standing cleanup authorization
applies only to kristine-named resources. The corrected launch retry remains
pending that response. No Qwen decision or world turn has completed on this TPU.
